from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch

from .base_method import Base_method
from .utils import cuda
from opencpd.models import ESMIF_Model
from opencpd.datasets.featurizer import featurize_Inversefolding


class ESMIF(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)

    def _build_model(self):
        return ESMIF_Model(self.args).to(self.device)
    
    def forward_loss(self, batch):
        X, S, score, mask, lengths, chain_mask = batch['X'], batch['S'], batch['score'], batch['mask'], batch['lengths'], batch['chain_mask']
        mask = mask==1
        S_prev = torch.clone(S).long()
        R = torch.rand(S_prev.shape[0])
        for i in range(R.shape[0]):
            S_prev[i, (R[i]*mask[i].sum()).long():] = 32
        
        log_probs, _ = self.model(X, mask, score, S_prev) 
        loss = self.criterion(log_probs, S)
        return {"loss":loss, 
                "S":S, 
                "log_probs":log_probs, 
                "chain_mask": chain_mask}

    def train_one_epoch(self, train_loader):
        self.model.train()
        train_sum, train_weights = 0., 0.

        train_pbar = tqdm(train_loader)
        for batch in train_pbar:
            self.optimizer.zero_grad()
            result = self.forward_loss(batch)
            loss, S, log_probs = result['loss'], result['S'], result['log_probs']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()

            nll_loss, _ = self.loss_nll_flatten(S, log_probs)
            mask = torch.ones_like(nll_loss)
            train_sum += torch.sum(nll_loss * mask).cpu().data.numpy()
            train_weights += torch.sum(mask).cpu().data.numpy()
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
                loss, S, log_probs = result['loss'], result['S'], result['log_probs']
                mask = torch.ones_like(loss)
                valid_sum += torch.sum(loss * mask).cpu().data.numpy()
                valid_weights += torch.sum(mask).cpu().data.numpy()

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
                loss, S, log_probs = result['loss'], result['S'], result['log_probs']
                mask = torch.ones_like(loss)
                test_sum += torch.sum(loss * mask).cpu().data.numpy()
                test_weights += torch.sum(mask).cpu().data.numpy()
                test_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))
            
            test_recovery, test_subcat_recovery = self._cal_recovery(test_loader.dataset, featurize_Inversefolding)
            
        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)
    
        return test_perplexity, test_recovery

    def _cal_recovery(self, dataset, featurizer):
        recovery = []
        subcat_recovery = {}
        with torch.no_grad():
            for protein in tqdm(dataset):
                p_category = protein['category'] if 'category' in protein.keys() else 'Unknown'
                if p_category not in subcat_recovery.keys():
                    subcat_recovery[p_category] = []

                batch = featurizer([protein])
                X, S, score, mask, lengths, chain_mask = batch['X'], batch['S'], batch['score'], batch['mask'], batch['lengths'], batch['chain_mask']
                X, S, score, mask, lengths, chain_mask = cuda([X, S, score, mask, lengths, chain_mask])
                mask = mask==1

                S_pred = self.model.sample(X, mask, confidence=score, temperature=0.1)
                cmp = (S_pred.view(1,-1) == S)
                cmp = cmp[score >= self.args.score_thr]
                recovery_ = cmp.float().mean().cpu().numpy()

                if np.isnan(recovery_): recovery_ = 0.0

                subcat_recovery[p_category].append(recovery_)
                recovery.append(recovery_)
            
            for key in subcat_recovery.keys():
                subcat_recovery[key] = np.median(subcat_recovery[key])

        recovery = np.median(recovery)
        return recovery, subcat_recovery

    def loss_nll_flatten(self, S, log_probs):
        """ Negative log probabilities """
        criterion = torch.nn.NLLLoss(reduction='none')
        loss = criterion(log_probs, S)
        loss_av = loss.mean()
        return loss, loss_av