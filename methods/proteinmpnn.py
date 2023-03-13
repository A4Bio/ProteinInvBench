import torch
import numpy as np
from tqdm import tqdm

from .structgnn import StructGNN_Design
from .utils import cuda, loss_nll, loss_smoothed
from models import ProteinMPNN_Model


class ProteinMPNN_Design(StructGNN_Design):
    def __init__(self, args, device, steps_per_epoch):
        StructGNN_Design.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)

        self.pssm_log_odds_flag = 0
        self.pssm_multi = 0.0
        self.pssm_bias_flag = 0
        self.pssm_threshold = 0.0

    def _build_model(self):
        return ProteinMPNN_Model(self.args).to(self.device)

    def _cal_recovery(self, dataset, featurizer):
        omit_AAs_list = 'X'
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
        bias_AAs_np = np.zeros(len(alphabet))

        recovery = []
        subcat_recovery = {}
        for protein in tqdm(dataset):
            p_category = protein['category'] if 'category' in protein.keys() else 'Unknown'
            if p_category not in subcat_recovery.keys():
                subcat_recovery[p_category] = []

            X, S, score, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = featurizer([protein], is_testing=True)
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_M_pos, omit_AA_mask, residue_idx, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all = cuda((X, S, mask, lengths, chain_M, chain_encoding_all, chain_M_pos, omit_AA_mask, residue_idx, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all), device=self.device)

            randn_2 = torch.randn(chain_M.shape, device=X.device)
            pssm_log_odds_mask = (pssm_log_odds_all > self.pssm_threshold).float() #1.0 for true, 0.0 for false

            sample_dict = self.model.sample(X, randn_2, torch.zeros_like(S), chain_M, chain_encoding_all, residue_idx, mask=mask, chain_M_pos=chain_M_pos, temperature=0.1, omit_AAs_np=omit_AAs_np, omit_AA_mask=omit_AA_mask, bias_AAs_np=bias_AAs_np, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=self.pssm_multi, pssm_log_odds_flag=bool(self.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(self.pssm_bias_flag), bias_by_res=bias_by_res_all)
            # sample_dict = self.model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, chain_M_pos=chain_M_pos, temperature=0.1, omit_AAs_np=omit_AAs_np, omit_AA_mask=omit_AA_mask, bias_AAs_np=bias_AAs_np, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=self.pssm_multi, pssm_log_odds_flag=bool(self.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(self.pssm_bias_flag), bias_by_res=bias_by_res_all)
            sample = sample_dict['S']

            cmp = sample.eq(S)
            cmp = cmp[score >= self.args.score_thr]
            recovery_ = cmp.float().mean().cpu().numpy()

            if np.isnan(recovery_): recovery_ = 0.0

            subcat_recovery[p_category].append(recovery_)
            recovery.append(recovery_)

        for key in subcat_recovery.keys():
            subcat_recovery[key] = np.median(subcat_recovery[key])

        self.mean_recovery = np.mean(recovery)
        self.std_recovery = np.std(recovery)
        self.min_recovery = np.min(recovery)
        self.max_recovery = np.max(recovery)
        self.median_recovery = np.median(recovery)
        recovery = np.median(recovery)
        return recovery, subcat_recovery

    def train_one_epoch(self, train_loader):
        """ train one epoch to obtain average loss """
        # Initialize the model
        self.model.train()
        train_sum, train_weights = 0., 0.
        # Start loading and training
        train_pbar = tqdm(train_loader)
        for batch in train_pbar:
            self.optimizer.zero_grad()
            X, S, _, mask, lengths, chain_M, chain_M_pos, residue_idx, chain_encoding_all = batch
            X, S, mask, lengths, chain_M, chain_M_pos, residue_idx, chain_encoding_all = cuda((X, S, mask, lengths, chain_M, chain_M_pos, residue_idx, chain_encoding_all), device=self.device)

            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = self.model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            mask_for_loss = mask * chain_M * chain_M_pos
            _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss, num_classes=self.args.vocab, weight=self.args.smoothing)
            loss_av_smoothed.backward()
        
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()

            loss, _ = loss_nll(S, log_probs, mask_for_loss)
            train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            train_pbar.set_description('train loss: {:.4f}'.format(loss.mean().item()))
        
        train_loss = train_sum / train_weights
        train_perplexity = np.exp(train_loss)
        return train_loss, train_perplexity
    
    def valid_one_epoch(self, valid_loader):
        self.model.eval()
        with torch.no_grad():
            valid_sum, valid_weights = 0., 0.
            valid_pbar = tqdm(valid_loader)
            for batch in valid_pbar:
                X, S, _, mask, lengths, chain_M, chain_M_pos, residue_idx, chain_encoding_all = batch
                X, S, mask, lengths, chain_M, chain_M_pos, residue_idx, chain_encoding_all = cuda((X, S, mask, lengths, chain_M, chain_M_pos, residue_idx, chain_encoding_all), device=self.device)

                randn_1 = torch.randn(chain_M.shape, device=X.device)
                log_probs = self.model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                mask_for_loss = mask * chain_M * chain_M_pos
                loss, _ = loss_nll(S, log_probs, mask_for_loss)

                valid_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                valid_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                valid_pbar.set_description('valid loss: {:.4f}'.format(loss.mean().item()))
        
        valid_loss = valid_sum / valid_weights
        valid_perplexity = np.exp(valid_loss)        
        return valid_loss, valid_perplexity

    def test_one_epoch(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            test_sum, test_weights = 0., 0.
            test_pbar = tqdm(test_loader)
            for batch in test_pbar:
                X, S, _, mask, lengths, chain_M, chain_M_pos, residue_idx, chain_encoding_all = batch
                X, S, mask, lengths, chain_M, chain_M_pos, residue_idx, chain_encoding_all = cuda((X, S, mask, lengths, chain_M, chain_M_pos, residue_idx, chain_encoding_all), device=self.device)

                randn_1 = torch.randn(chain_M.shape, device=X.device)
                log_probs = self.model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                mask_for_loss = mask * chain_M * chain_M_pos
                loss, _ = loss_nll(S, log_probs, mask_for_loss)
                
                test_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                test_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                test_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))

            test_recovery, test_subcat_recovery = self._cal_recovery(test_loader.dataset, test_loader.featurizer)
            
        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)
        return test_perplexity, test_recovery, test_subcat_recovery