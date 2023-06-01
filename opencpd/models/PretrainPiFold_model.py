import time
import torch
import math
import torch.nn as nn
from opencpd.utils import gather_nodes, _dihedrals, _get_rbf, _get_dist, _rbf, _orientations_coarse_gl_tuple
from opencpd.modules.pifold_module import StructureEncoder, MLPDecoder
from transformers import AutoTokenizer
import numpy as np
import torch.nn.functional as F
import os.path as osp

pair_lst = ['N-N', 'C-C', 'O-O', 'Cb-Cb', 'Ca-N', 'Ca-C', 'Ca-O', 'Ca-Cb', 'N-C', 'N-O', 'N-Cb', 'Cb-C', 'Cb-O', 'O-C', 'N-Ca', 'C-Ca', 'O-Ca', 'Cb-Ca', 'C-N', 'O-N', 'Cb-N', 'C-Cb', 'O-Cb', 'C-O']


class PretrainPiFold_Model(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(PretrainPiFold_Model, self).__init__()
        self.args = args
        self.augment_eps = args.augment_eps
        node_features = args.node_features
        edge_features = args.edge_features
        hidden_dim = args.hidden_dim
        dropout = args.dropout
        num_encoder_layers = args.num_encoder_layers
        self.top_k = args.k_neighbors
        self.args.virtual_num = 3
        
        self.num_rbf = 16
        self.num_positional_embeddings = 16

        self.dihedral_type = args.dihedral_type
        
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers")
        alphabet = [one for one in 'ACDEFGHIKLMNPQRSTVWYX']
        self.token_mask = torch.tensor([(one in alphabet) for one in self.tokenizer._token_to_id.keys()])
        # self.full_atom_dis = args.full_atom_dis
        
        # node_in = 12

        # node_in = node_in + 9 + 576 # node_in + 9 + 576
        prior_matrix = [
            [-0.58273431, 0.56802827, -0.54067466],
            [0.0       ,  0.83867057, -0.54463904],
            [0.01984028, -0.78380804, -0.54183614],
        ]
        

        # prior_matrix = torch.rand(self.args.virtual_num,3)
        self.virtual_atoms = nn.Parameter(torch.tensor(prior_matrix)[:self.args.virtual_num,:])

        node_in = 0
        if self.args.node_dist:
            pair_num = 6
            if self.args.virtual_num>0:
                pair_num += self.args.virtual_num*(self.args.virtual_num-1)
            node_in += pair_num*self.num_rbf
        if self.args.node_angle:
            node_in += 12
        if self.args.node_direct:
            node_in += 9
        
        edge_in = 0
        if self.args.edge_dist:
            pair_num = 0
            if self.args.Ca_Ca:
                pair_num += 1
            if self.args.Ca_C:
                pair_num += 2
            if self.args.Ca_N:
                pair_num += 2
            if self.args.Ca_O:
                pair_num += 2
            if self.args.C_C:
                pair_num += 1
            if self.args.C_N:
                pair_num += 2
            if self.args.C_O:
                pair_num += 2
            if self.args.N_N:
                pair_num += 1
            if self.args.N_O:
                pair_num += 2
            if self.args.O_O:
                pair_num += 1

            
            if self.args.virtual_num>0:
                pair_num += self.args.virtual_num
                pair_num += self.args.virtual_num*(self.args.virtual_num-1)
            edge_in += pair_num*self.num_rbf
        if self.args.edge_angle:
            edge_in += 4
        if self.args.edge_direct:
            edge_in += 12
        
        # edge_in += 16 # position encoding
        
        if self.args.use_gvp_feat:
            node_in = 12
            edge_in = 48-16

        edge_in += 16 # position encoding
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = nn.BatchNorm1d(node_features)
        self.norm_edges = nn.BatchNorm1d(edge_features)

        self.W_v = nn.Sequential(
            nn.Linear(node_features, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )
        
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True) 
        self.W_f = nn.Linear(edge_features, hidden_dim, bias=True)

        self.encoder = StructureEncoder(hidden_dim, num_encoder_layers, dropout, args.updating_edges, args.att_output_mlp, args.node_output_mlp, args.node_net, args.edge_net, args.node_context, args.edge_context)

        # self.decoder = CNNDecoder(hidden_dim, hidden_dim, args.num_decoder_layers1, args.kernel_size1, args.act_type, args.glu)
        # self.decoder2 = CNNDecoder2(hidden_dim, hidden_dim, args.num_decoder_layers2, args.kernel_size2, args.act_type, args.glu)


        self.decoder = MLPDecoder(hidden_dim, hidden_dim, args.num_decoder_layers1, args.kernel_size1, args.act_type, args.glu, vocab=len(self.tokenizer._token_to_id))
        self._init_params()

        # self.load_state_dict(torch.load("/gaozhangyang/experiments/PiFoldV2/results/cath4.2_pretrained/checkpoint.pth"))
        # self.load_state_dict(torch.load("/gaozhangyang/experiments/PiFoldV2/results/PiFold_esm_token/checkpoint.pth"))
        pretrain_pifold_path = osp.join(self.args.res_dir, self.data_name, "PiFold", "checkpoint.pth")
        self.load_state_dict(torch.load(pretrain_pifold_path))
    
    @torch.no_grad()
    def forward(self, batch):
        h_V, h_P, P_idx, batch_id = batch['h_V'], batch['h_E'], batch['E_idx'], batch['batch_id']
        device = h_V.device
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))
        
        h_V, h_P = self.encoder(h_V, h_P, P_idx, batch_id)
        log_probs, logits = self.decoder(h_V, batch_id)
        probs = F.softmax(logits, dim=-1)
        conf, pred_id = probs.max(dim=-1)
        
        maxL = 0
        for b in batch_id.unique():
            mask = batch_id==b
            L = mask.sum()
            if L>maxL:
                maxL=L
        
        confs = []
        seqs = []
        embeds = []
        probs2 = []
        for b in batch_id.unique():
            mask = batch_id==b
            # elements = [alphabet[int(id)] for id in pred_id[mask]]
            elements = self.tokenizer.decode(pred_id[mask]).split(" ")
            seqs.append(elements)
            confs.append(conf[mask])
            embeds.append(h_V[mask])
            probs2.append(probs[mask])
        
        seqs = self.tokenizer(["".join(one) for one in seqs], padding=True, truncation=True, return_tensors='pt', add_special_tokens=False)
        confs = torch.stack([F.pad(one, (0, maxL-len(one))) for one in confs])
        embeds = torch.stack([F.pad(one, (0,0, 0, maxL-len(one))) for one in embeds])
        probs2 = torch.stack([F.pad(one, (0,0, 0, maxL-len(one)), value=1/33) for one in probs2])
        
        ret = {"pred_ids":seqs['input_ids'].to(device),
               "confs":confs,
               "embeds":embeds,
               "probs":probs2,
               "attention_mask":seqs['attention_mask'].to(device),
               "E_idx":P_idx,
               "batch_id":batch_id,
               "h_E":h_P}
        return ret
        
    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx  

    def _get_features(self, S, score, X, mask, chain_mask, chain_encoding):
        if self.augment_eps>0:
            X = X + self.augment_eps * torch.randn_like(X)
        
        device = X.device
        mask_bool = (mask==1)
        B, N, _,_ = X.shape
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx = self._full_dist(X_ca, mask, self.top_k)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x:  torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1,x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])



        # sequence
        S = torch.masked_select(S, mask_bool)
        if score is not None:
            score = torch.masked_select(score, mask_bool)
        chain_mask = torch.masked_select(chain_mask, mask_bool)
        chain_encoding = torch.masked_select(chain_encoding, mask_bool)

        # angle & direction
        V_angles = _dihedrals(X, self.dihedral_type) 
        V_angles = node_mask_select(V_angles)

        V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)
        V_direct = node_mask_select(V_direct)
        E_direct = edge_mask_select(E_direct)
        E_angles = edge_mask_select(E_angles)

        # distance
        atom_N = X[:,:,0,:]
        atom_Ca = X[:,:,1,:]
        atom_C = X[:,:,2,:]
        atom_O = X[:,:,3,:]
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = torch.cross(b, c, dim=-1)

        if self.args.virtual_num>0:
            virtual_atoms = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
            for i in range(self.virtual_atoms.shape[0]):
                vars()['atom_v' + str(i)] = virtual_atoms[i][0] * a \
                                        + virtual_atoms[i][1] * b \
                                        + virtual_atoms[i][2] * c \
                                        + 1 * atom_Ca

        node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            node_dist.append( node_mask_select(_get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze()))
        
        if self.args.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                for j in range(0, i):
                    node_dist.append(node_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], None, self.num_rbf).squeeze()))
                    node_dist.append(node_mask_select(_get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], None, self.num_rbf).squeeze()))
        V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()
        
        

        pair_lst = []
        if self.args.Ca_Ca:
            pair_lst.append('Ca-Ca')
        if self.args.Ca_C:
            pair_lst.append('Ca-C')
            pair_lst.append('C-Ca')
        if self.args.Ca_N:
            pair_lst.append('Ca-N')
            pair_lst.append('N-Ca')
        if self.args.Ca_O:
            pair_lst.append('Ca-O')
            pair_lst.append('O-Ca')
        if self.args.C_C:
            pair_lst.append('C-C')
        if self.args.C_N:
            pair_lst.append('C-N')
            pair_lst.append('N-C')
        if self.args.C_O:
            pair_lst.append('C-O')
            pair_lst.append('O-C')
        if self.args.N_N:
            pair_lst.append('N-N')
        if self.args.N_O:
            pair_lst.append('N-O')
            pair_lst.append('O-N')
        if self.args.O_O:
            pair_lst.append('O-O')

        
        edge_dist = [] #Ca-Ca
        for pair in pair_lst:
            atom1, atom2 = pair.split('-')
            rbf = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, self.num_rbf)
            edge_dist.append(edge_mask_select(rbf))

        if self.args.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(i)], E_idx, self.num_rbf)))

                for j in range(0, i):
                    edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], E_idx, self.num_rbf)))
                    edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], E_idx, self.num_rbf)))

        
        E_dist = torch.cat(tuple(edge_dist), dim=-1)

        h_V = []
        if self.args.node_dist:
            h_V.append(V_dist)
        if self.args.node_angle:
            h_V.append(V_angles)
        if self.args.node_direct:
            h_V.append(V_direct)
        
        h_E = []
        if self.args.edge_dist:
            h_E.append(E_dist)
        if self.args.edge_angle:
            h_E.append(E_angles)
        if self.args.edge_direct:
            h_E.append(E_direct)
        
        _V = torch.cat(h_V, dim=-1)
        _E = torch.cat(h_E, dim=-1)

        
    
        
        # edge index
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B,1,1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1,-1)
        dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1,-1)
        E_idx = torch.cat((dst, src), dim=0).long()
        
        pos_embed = self._positional_embeddings(E_idx, 16)
        _E = torch.cat([_E, pos_embed], dim=-1)


        
        # 3D point
        sparse_idx = mask.nonzero()  # index of non-zero values
        X = X[sparse_idx[:,0], sparse_idx[:,1], :, :]
        batch_id = sparse_idx[:,0]

        return X, S, score, _V, _E, E_idx, batch_id, chain_mask, chain_encoding

    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E
        
        
    
    def _get_features_analysis(self, S, score, X, mask):
        device = X.device
        mask_bool = (mask==1)
        B, N, _,_ = X.shape
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx = self._full_dist(X_ca, mask, self.top_k)
        E_idx = E_idx[:,:,1:2]

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x:  torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1,x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])

        # sequence
        S = torch.masked_select(S, mask_bool)
        if score is not None:
            score = torch.masked_select(score, mask_bool)

        # angle & direction
        V_angles_names = ['psi', 'omega', 'phi', 'alpha', 'beta', 'gamma']
        V_angles = _dihedrals(X, self.dihedral_type, return_raw=True) 
        V_angles = node_mask_select(V_angles)

        V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)
        V_direct = node_mask_select(V_direct)
        E_direct = edge_mask_select(E_direct)
        E_angles = edge_mask_select(E_angles)

        # distance
        atom_N = X[:,:,0,:]
        atom_Ca = X[:,:,1,:]
        atom_C = X[:,:,2,:]
        atom_O = X[:,:,3,:]
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = torch.cross(b, c, dim=-1)

        if self.args.virtual_num>0:
            virtual_atoms = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
            for i in range(self.virtual_atoms.shape[0]):
                vars()['atom_v' + str(i)] = virtual_atoms[i][0] * a \
                                        + virtual_atoms[i][1] * b \
                                        + virtual_atoms[i][2] * c \
                                        + 1 * atom_Ca

        V_dist_names = []
        node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            node_dist.append( node_mask_select(_get_dist(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf)))
            V_dist_names.append(pair)
        
        if self.args.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                for j in range(0, i):
                    node_dist.append(node_mask_select(_get_dist(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], None, self.num_rbf)))
                    node_dist.append(node_mask_select(_get_dist(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], None, self.num_rbf)))
                    V_dist_names.append('{}-{}'.format('atom_v' + str(i), 'atom_v' + str(j)))
                    V_dist_names.append('{}-{}'.format('atom_v' + str(j), 'atom_v' + str(i)))
        V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()

        E_dist_names = []
        pair_lst = []
        if self.args.Ca_Ca:
            pair_lst.append('Ca-Ca')
        if self.args.Ca_C:
            pair_lst.append('Ca-C')
            pair_lst.append('C-Ca')
        if self.args.Ca_N:
            pair_lst.append('Ca-N')
            pair_lst.append('N-Ca')
        if self.args.Ca_O:
            pair_lst.append('Ca-O')
            pair_lst.append('O-Ca')
        if self.args.C_C:
            pair_lst.append('C-C')
        if self.args.C_N:
            pair_lst.append('C-N')
            pair_lst.append('N-C')
        if self.args.C_O:
            pair_lst.append('C-O')
            pair_lst.append('O-C')
        if self.args.N_N:
            pair_lst.append('N-N')
        if self.args.N_O:
            pair_lst.append('N-O')
            pair_lst.append('O-N')
        if self.args.O_O:
            pair_lst.append('O-O')

        
        edge_dist = [] #Ca-Ca
        for pair in pair_lst:
            atom1, atom2 = pair.split('-')
            rbf = _get_dist(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, self.num_rbf).unsqueeze(-1)
            edge_dist.append(edge_mask_select(rbf))
            E_dist_names.append(pair)

        if self.args.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                edge_dist.append(edge_mask_select(_get_dist(vars()['atom_v' + str(i)], vars()['atom_v' + str(i)], E_idx, self.num_rbf).unsqueeze(-1)))
                E_dist_names.append('{}-{}'.format('atom_v' + str(i), 'atom_v' + str(i)))
                for j in range(0, i):
                    edge_dist.append(edge_mask_select(_get_dist(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], E_idx, self.num_rbf).unsqueeze(-1)))
                    edge_dist.append(edge_mask_select(_get_dist(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], E_idx, self.num_rbf).unsqueeze(-1)))
                    E_dist_names.append('{}-{}'.format('atom_v' + str(i), 'atom_v' + str(j)))
                    E_dist_names.append('{}-{}'.format('atom_v' + str(j), 'atom_v' + str(i)))

        
        E_dist = torch.cat(tuple(edge_dist), dim=-1)

        return V_dist, V_angles, V_direct, E_dist, E_angles, E_direct, V_dist_names, V_angles_names, E_dist_names