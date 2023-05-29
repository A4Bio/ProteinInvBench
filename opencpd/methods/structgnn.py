import torch
import numpy as np
from tqdm import tqdm

from .base_method import Base_method
from .utils import cuda, loss_nll, loss_smoothed
from opencpd.models import StructGNN_Model
import torch.nn as nn
from opencpd.datasets.featurizer import featurize_GTrans

class StructGNN(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def _build_model(self):
        return StructGNN_Model(self.args).to(self.device)

    def _cal_recovery(self, dataset, featurizer):
        recovery = []
        subcat_recovery = {}
        for protein in tqdm(dataset):
            p_category = protein['category'] if 'category' in protein.keys() else 'Unknown'
            if p_category not in subcat_recovery.keys():
                subcat_recovery[p_category] = []

            batch = featurizer([protein])
            if self.args.method == 'GCA':
                X, S, score, mask, lengths, chain_mask, chain_encoding = batch['X'], batch['S'], batch['score'], batch['mask'], batch['lengths'], batch['chain_mask'], batch['chain_encoding']
                X, S, score, mask, lengths, chain_mask, chain_encoding = cuda([X, S, score, mask, lengths, chain_mask, chain_encoding])
                
                h_V, h_P, h_F, P_idx, F_idx, chain_mask = self.model._get_features(X, lengths, mask, chain_mask=chain_mask, chain_encoding = chain_encoding)
                
                sample = self.model.sample(h_V, h_P, h_F, P_idx, F_idx, mask, temperature=0.1)
            else:
                X, S, score, mask, lengths, chain_mask, chain_encoding = batch['X'], batch['S'], batch['score'], batch['mask'], batch['lengths'], batch['chain_mask'], batch['chain_encoding']
                X, S, score, mask, lengths, chain_mask, chain_encoding = cuda([X, S, score, mask, lengths, chain_mask, chain_encoding])
                
                V, E, E_idx, chain_mask = self.model._get_features(X, lengths, mask, chain_mask=chain_mask, chain_encoding = chain_encoding)
                
                sample = self.model.sample(V, E, E_idx, mask, temperature=0.1)
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
        recovery = self.median_recovery
        return recovery, subcat_recovery
    
    def forward_loss(self, batch):
        X, S, score, mask, lengths = batch['X'], batch['S'], batch['score'], batch['mask'], batch['lengths']
        if self.args.method=='GCA':
            h_V, h_P, h_F, P_idx, F_idx, chain_mask = self.model._get_features(X, lengths, mask, chain_mask=batch['chain_mask'], chain_encoding = batch['chain_encoding'])
            
            log_probs = self.model(h_V, h_P, h_F, P_idx, F_idx, S, mask)
        else:
            V, E, E_idx, chain_mask = self.model._get_features(X, lengths, mask, chain_mask=batch['chain_mask'], chain_encoding = batch['chain_encoding'])
            log_probs = self.model(S, V, E, E_idx, mask)
        B, L = S.shape
        loss = self.criterion(log_probs.reshape(B*L, -1), S.reshape(B*L)).reshape(B,L)
        loss = (loss*mask).sum()/(mask.sum())
        # _, loss = loss_smoothed(S, log_probs, mask, weight=self.args.smoothing, num_classes=33)
        return {"S":S, 
                "log_probs":log_probs,
                "mask": mask,
                "loss": loss}
        
        

    def train_one_epoch(self, train_loader):
        """ train one epoch to obtain average loss """
        # Initialize the model
        self.model.train()
        train_sum, train_weights = 0., 0.
        # Start loading and training
        train_pbar = tqdm(train_loader)
        for batch in train_pbar:
            self.optimizer.zero_grad()
            result = self.forward_loss(batch)
            S, log_probs, mask, loss = result['S'], result['log_probs'], result['mask'], result['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()

            train_sum += torch.sum(loss * mask).cpu().data.numpy()
            train_weights += torch.sum(mask).cpu().data.numpy()
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
                result = self.forward_loss(batch)
                S, log_probs, mask = result['S'], result['log_probs'], result['mask']
                loss, _ = loss_nll(S, log_probs, mask)

                valid_sum += torch.sum(loss * mask).cpu().data.numpy()
                valid_weights += torch.sum(mask).cpu().data.numpy()

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
                result = self.forward_loss(batch)
                S, log_probs, mask = result['S'], result['log_probs'], result['mask']
                loss, _ = loss_nll(S, log_probs, mask)
                
                test_sum += torch.sum(loss * mask).cpu().data.numpy()
                test_weights += torch.sum(mask).cpu().data.numpy()

                test_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))

            test_recovery, test_subcat_recovery = self._cal_recovery(test_loader.dataset, featurize_GTrans)
            
        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)
        return test_perplexity, test_recovery