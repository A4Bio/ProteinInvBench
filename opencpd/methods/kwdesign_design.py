from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch

from .base_method import Base_method
from .utils import cuda
from opencpd.models import KWDesign_model
from opencpd.datasets.featurizer import featurize_GTrans
import os.path as osp
from opencpd.utils.main_utils import check_dir
import os

class KWDesign(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.args = args
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        if self.args.load_memory:
            sv_root = osp.join(self.args.res_dir, self.args.data_name)
            check_dir(sv_root)
            self.args.memory_path = osp.join(sv_root, 'memory.pth')
            if os.path.exists(self.args.memory_path):
                memories = torch.load(self.args.memory_path)
                self.model.memo_pifold.memory = memories['memo_pifold']
                self.model.memo_esmif.memory = memories['memo_esmif']
        
        if self.args.recycle_n>1:
            params = torch.load(osp.join(self.args.res_dir, self.args.ex_name, "checkpoints", f"msa{self.args.msa_n}_recycle{self.args.recycle_n-1}_epoch{self.args.load_epoch}.pth"))
            self.model.load_state_dict(params, strict=False)
            for i in range(self.args.recycle_n-1):
                submodule = self.model.get_submodule(f"Design{i+1}")
                submodule.fix_memory = True
                for p in submodule.parameters():
                    p.requires_grad = False
                    
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.stop = False
        
    def _build_model(self):
        return KWDesign_model(self.args).to(self.device)
    
    
    def forward_loss(self,batch):
        X, S, score, mask, lengths = batch['X'], batch['S'], batch['score'], batch['mask'], batch['lengths']

        X, S, score, h_V, h_E, E_idx, batch_id, chain_mask, chain_encoding= self.model.Design1.design_model.PretrainPiFold._get_features(S, score, X=X, mask=mask, chain_mask=batch['chain_mask'], chain_encoding = batch['chain_encoding'])
        

        batch = {"title": batch['title'], "h_V":h_V, "h_E":h_E, "E_idx":E_idx, "batch_id":batch_id, "alphabet":'ACDEFGHIKLMNPQRSTVWYX', 'S':S, 'position':X}
        
        results = self.model(batch)
        loss = 0
        for i in range(1, self.args.recycle_n+1):
            loss += self.criterion(results[f'log_probs{i}'], S)
                    
        loss = (loss*chain_mask).sum()/chain_mask.sum()/5
        return {"loss":loss, 
                "S":S, 
                "log_probs":results['log_probs'], 
                "chain_mask": chain_mask}
    
    def get_metric(self, S, log_probs, chain_mask):
        nll_loss, _ = self.loss_nll_flatten(S, log_probs)
        loss = torch.sum(nll_loss * chain_mask).cpu().data.numpy()
        weight = torch.sum(chain_mask).cpu().data.numpy()
        return {"loss":loss, "weight":weight}

    def train_one_epoch(self, train_loader):
        self.model.train()
        train_sum, train_weights = 0., 0.

        train_pbar = tqdm(train_loader)
        for step_idx, batch in enumerate(train_pbar):
            self.optimizer.zero_grad()
            result = self.forward_loss(batch)
            loss = result['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()
            
            
            metric = self.get_metric(result['S'], result['log_probs'], result['chain_mask'])

            train_sum += metric['loss']
            train_weights += metric['weight']
            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
            
            torch.cuda.empty_cache()
        
        train_loss = train_sum / train_weights
        train_perplexity = np.exp(train_loss)
        return train_loss, train_perplexity

    def valid_one_epoch(self, valid_loader):
        self.model.eval()
        valid_sum, valid_weights = 0., 0.
        valid_pbar = tqdm(valid_loader)

        with torch.no_grad():
            for batch in valid_pbar:

                result = self.forward_loss(batch)
                loss = result['loss']
                metric = self.get_metric(result['S'], result['log_probs'], result['chain_mask'])

                valid_sum += metric['loss']
                valid_weights += metric['weight']

                valid_pbar.set_description('valid loss: {:.4f}'.format(loss.mean().item()))
                
            valid_loss = valid_sum / valid_weights
            valid_perplexity = np.exp(valid_loss)
        return valid_loss, valid_perplexity

    def test_one_epoch(self, test_loader):
        self.model.eval()
        test_sum, test_weights = 0., 0.
        test_pbar = tqdm(test_loader)

        with torch.no_grad():
            for batch in test_pbar:
                result = self.forward_loss(batch)
                loss = result['loss']
                metric = self.get_metric(result['S'], result['log_probs'], result['chain_mask'])
                
                test_sum += metric['loss']
                test_weights += metric['weight']
                test_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))
            
            test_recovery = self._cal_recovery(test_loader.dataset, featurize_GTrans)
            
        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)
    
        return test_perplexity, test_recovery

    def _cal_recovery(self, dataset, featurizer):
        recovery = []
        subcat_recovery = {}
        pred_results = {}
        with torch.no_grad():
            for protein in tqdm(dataset):
                if protein is None:
                    continue
                name = protein['title']
                p_category = protein['category'] if 'category' in protein.keys() else 'Unknown'
                if p_category not in subcat_recovery.keys():
                    subcat_recovery[p_category] = []
                    
                    
                protein = featurizer([protein])
                tocuda = lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x
                protein = {key:tocuda(val) for key, val in protein.items()}
                X, S, mask, score, lengths, chain_mask = protein['X'], protein['S'], protein['mask'], protein['score'], protein['lengths'], protein['chain_mask']
                
                X, S, score, h_V, h_E, E_idx, batch_id, chain_mask, chain_encoding= self.model.Design1.design_model.PretrainPiFold._get_features(S, score, X=X, mask=mask, chain_mask=protein['chain_mask'], chain_encoding = protein['chain_encoding'])
                
                
                batch = {"title": protein['title'],"h_V":h_V, "h_E":h_E, "E_idx":E_idx, "batch_id":batch_id, "alphabet":'ACDEFGHIKLMNPQRSTVWYX', "S":S, 'position':X}
                    
                results = self.model(batch)
                log_probs = results['log_probs']
                S_pred = torch.argmax(log_probs, dim=1)
                cmp = (S_pred == S)
                # cmp = cmp[score >= self.args.score_thr]
                recovery_ = ((cmp.float()*chain_mask).sum()/chain_mask.sum()).cpu().numpy()

                if np.isnan(recovery_): recovery_ = 0.0
                
                subcat_recovery[p_category].append(recovery_)
                recovery.append(recovery_)

                recovery.append(recovery_)
                
                pred_seq = "".join(self.model.Design1.design_model.PretrainPiFold.tokenizer.decode(S_pred).split(" "))
                true_seq = "".join(self.model.Design1.design_model.PretrainPiFold.tokenizer.decode(S).split(" "))
                
                pred_results[name] = {"seq":pred_seq,
                                      "prob": torch.exp(log_probs).cpu().numpy().tolist()}
            
            for key in subcat_recovery.keys():
                subcat_recovery[key] = np.median(subcat_recovery[key])
        
        # import json
        # json.dump(pred_results, open("./pifold2_results.json", "w"))
        self.mean_recovery = np.mean(recovery)
        self.std_recovery = np.std(recovery)
        self.min_recovery = np.min(recovery)
        self.max_recovery = np.max(recovery)
        self.median_recovery = np.median(recovery)
        recovery = np.median(recovery)
        return recovery

    def loss_nll_flatten(self, S, log_probs):
        """ Negative log probabilities """
        criterion = torch.nn.NLLLoss(reduction='none')
        loss = criterion(log_probs, S)
        loss_av = loss.mean()
        return loss, loss_av
    


class PiFoldV2(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.args = args
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)

    def _build_model(self):
        return PiFoldV2_Model(self.args).to(self.device)
    
    
    def forward_loss(self,batch):
        X, S, score, mask, lengths = batch['X'], batch['S'], batch['score'], batch['mask'], batch['lengths']

        X, S, score, h_V, h_E, E_idx, batch_id, chain_mask, chain_encoding= self.model.PretrainPiFold._get_features(S, score, X=X, mask=mask, chain_mask=batch['chain_mask'], chain_encoding = batch['chain_encoding'])
        

        batch = {"title": batch['title'], "h_V":h_V, "h_E":h_E, "E_idx":E_idx, "batch_id":batch_id, "alphabet":'ACDEFGHIKLMNPQRSTVWYX', 'S':S}
        
        log_probs = self.model(batch)
        loss = self.criterion(log_probs, S)
        loss = (loss*chain_mask).sum()/chain_mask.sum()
        return {"loss":loss, 
                "S":S, 
                "log_probs":log_probs, 
                "chain_mask": chain_mask}
    
    def get_metric(self, S, log_probs, chain_mask):
        nll_loss, _ = self.loss_nll_flatten(S, log_probs)
        loss = torch.sum(nll_loss * chain_mask).cpu().data.numpy()
        weight = torch.sum(chain_mask).cpu().data.numpy()
        return {"loss":loss, "weight":weight}

    def train_one_epoch(self, train_loader):
        self.model.train()
        train_sum, train_weights = 0., 0.

        train_pbar = tqdm(train_loader)
        for step_idx, batch in enumerate(train_pbar):
            self.optimizer.zero_grad()
            result = self.forward_loss(batch)
            loss = result['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()
            
            
            metric = self.get_metric(result['S'], result['log_probs'], result['chain_mask'])

            train_sum += metric['loss']
            train_weights += metric['weight']
            train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))
            
        train_loss = train_sum / train_weights
        train_perplexity = np.exp(train_loss)
        return train_loss, train_perplexity

    def valid_one_epoch(self, valid_loader):
        self.model.eval()
        valid_sum, valid_weights = 0., 0.
        valid_pbar = tqdm(valid_loader)

        with torch.no_grad():
            for batch in valid_pbar:

                result = self.forward_loss(batch)
                loss = result['loss']
                metric = self.get_metric(result['S'], result['log_probs'], result['chain_mask'])

                valid_sum += metric['loss']
                valid_weights += metric['weight']

                valid_pbar.set_description('valid loss: {:.4f}'.format(loss.mean().item()))
                
            valid_loss = valid_sum / valid_weights
            valid_perplexity = np.exp(valid_loss)
        return valid_loss, valid_perplexity

    def test_one_epoch(self, test_loader):
        self.model.eval()
        test_sum, test_weights = 0., 0.
        test_pbar = tqdm(test_loader)

        with torch.no_grad():
            for batch in test_pbar:
                result = self.forward_loss(batch)
                loss = result['loss']
                metric = self.get_metric(result['S'], result['log_probs'], result['chain_mask'])
                
                test_sum += metric['loss']
                test_weights += metric['weight']
                test_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))
            
            test_recovery = self._cal_recovery(test_loader.dataset, featurize_GTrans)
            
        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)
    
        return test_perplexity, test_recovery

    def _cal_recovery(self, dataset, featurizer):
        recovery = []
        subcat_recovery = {}
        pred_results = {}
        with torch.no_grad():
            for protein in tqdm(dataset):
                if protein is None:
                    continue
                name = protein['title']
                protein = featurizer([protein])
                tocuda = lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x
                protein = {key:tocuda(val) for key, val in protein.items()}
                X, S, mask, score, lengths, chain_mask = protein['X'], protein['S'], protein['mask'], protein['score'], protein['lengths'], protein['chain_mask']
                
                X, S, score, h_V, h_E, E_idx, batch_id, chain_mask, chain_encoding= self.model.PretrainPiFold._get_features(S, score, X=X, mask=mask, chain_mask=protein['chain_mask'], chain_encoding = protein['chain_encoding'])
                
                
                batch = {"title": protein['title'],"h_V":h_V, "h_E":h_E, "E_idx":E_idx, "batch_id":batch_id, "alphabet":'ACDEFGHIKLMNPQRSTVWYX', "S":S}
                    
                log_probs = self.model(batch)
                S_pred = torch.argmax(log_probs, dim=1)
                cmp = (S_pred == S)
                # cmp = cmp[score >= self.args.score_thr]
                recovery_ = ((cmp.float()*chain_mask).sum()/chain_mask.sum()).cpu().numpy()

                if np.isnan(recovery_): recovery_ = 0.0

                recovery.append(recovery_)
                
                pred_seq = "".join(self.model.tokenizer.decode(S_pred).split(" "))
                true_seq = "".join(self.model.tokenizer.decode(S).split(" "))
                print(true_seq)
                
                pred_results[name] = {"seq":pred_seq,
                                      "prob": torch.exp(log_probs).cpu().numpy().tolist()}
            
        # import json
        # json.dump(pred_results, open("./pifold2_results.json", "w"))
        self.mean_recovery = np.mean(recovery)
        self.std_recovery = np.std(recovery)
        self.min_recovery = np.min(recovery)
        self.max_recovery = np.max(recovery)
        self.median_recovery = np.median(recovery)
        recovery = np.median(recovery)
        return recovery

    def loss_nll_flatten(self, S, log_probs):
        """ Negative log probabilities """
        criterion = torch.nn.NLLLoss(reduction='none')
        loss = criterion(log_probs, S)
        loss_av = loss.mean()
        return loss, loss_av
    


