from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch

from .base_method import Base_method
from .utils import cuda
from opencpd.models import PiFold_Model
from opencpd.datasets.featurizer import featurize_GTrans

class PiFold(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer, self.scheduler = self._init_optimizer(steps_per_epoch)
        

    def _build_model(self):
        return PiFold_Model(self.args).to(self.device)
    
    
    def forward_loss(self,batch):
        X, S, score, mask, lengths = batch['X'], batch['S'], batch['score'], batch['mask'], batch['lengths']
        if self.args.augment_eps>0:
            X = X + self.args.augment_eps * torch.randn_like(X)
            
        X, S, score, h_V, h_E, E_idx, batch_id, chain_mask, chain_encoding= self.model._get_features(S, score, X=X, mask=mask, chain_mask=batch['chain_mask'], chain_encoding = batch['chain_encoding'])
        
        log_probs = self.model(h_V, h_E, E_idx, batch_id)
        loss = self.criterion(log_probs, S)
        loss = (loss*chain_mask).sum()/(chain_mask.sum())
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
            if torch.isnan(loss):
                print("nan at step {}".format(step_idx))
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
        true_results = {}
        with torch.no_grad():
            for protein in tqdm(dataset):
                if protein is None:
                    continue
                name = protein['title']
                protein = featurizer([protein])
                protein = cuda(protein)
                X, S, mask, score, lengths, chain_mask = protein['X'], protein['S'], protein['mask'], protein['score'], protein['lengths'], protein['chain_mask']

                X, S, score, h_V, h_E, E_idx, batch_id, chain_mask, chain_encoding= self.model._get_features(S, score, X=X, mask=mask, chain_mask=chain_mask, chain_encoding = protein['chain_encoding'])
                    
                log_probs = self.model(h_V, h_E, E_idx, batch_id)
                S_pred = torch.argmax(log_probs, dim=1)
                cmp = (S_pred == S)
                # cmp = cmp[score >= self.args.score_thr]
                recovery_ = ((cmp.float()*chain_mask).sum()/chain_mask.sum()).cpu().numpy()

                if np.isnan(recovery_): recovery_ = 0.0

                recovery.append(recovery_)
                
                
                pred_seq = "".join(self.model.tokenizer.decode(S_pred).split(" "))
                true_seq = "".join(self.model.tokenizer.decode(S).split(" "))
                # print(name)
                # print(pred_seq)
                # print(true_seq)
                
                
                pred_results[name] = {"seq":pred_seq,
                                      "prob": torch.exp(log_probs).cpu().numpy().tolist()}
                
                
                true_results[name] = {"seq": true_seq,
                                      "conf":None}
                
                
                

            
        # import json
        # json.dump(pred_results, open("./pifold_results.json", "w"))
        # json.dump(true_results, open("./true_results.json", "w"))
        self.mean_recovery = np.mean(recovery)
        self.std_recovery = np.std(recovery)
        self.min_recovery = np.min(recovery)
        self.max_recovery = np.max(recovery)
        self.median_recovery = np.median(recovery)
        recovery = np.median(recovery)
        return recovery
    
    
    def _save_probs(self, dataset, featurizer):
        from transformers import AutoTokenizer
        sv_results = {"title": [],
                      "true_seq":[],
                      "pred_probs":[],
                      "tokenizer":AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers")}
        with torch.no_grad():
            for protein in tqdm(dataset):
                if protein is None:
                    continue
                name = protein['title']
                protein = featurizer([protein])
                protein = cuda(protein)
                X, S, mask, score, lengths, chain_mask = protein['X'], protein['S'], protein['mask'], protein['score'], protein['lengths'], protein['chain_mask']
                
                if self.args.augment_eps>0:
                    X = X + self.args.augment_eps * torch.randn_like(X)

                X, S, score, h_V, h_E, E_idx, batch_id, chain_mask, chain_encoding= self.model._get_features(S, score, X=X, mask=mask, chain_mask=chain_mask, chain_encoding = protein['chain_encoding'])
                
                    
                log_probs = self.model(h_V, h_E, E_idx, batch_id)
                
                sv_results['title'].append(name)
                sv_results['true_seq'].append(S.cpu())
                sv_results['pred_probs'].append(torch.exp(log_probs).cpu())
        return sv_results

    def loss_nll_flatten(self, S, log_probs):
        """ Negative log probabilities """
        criterion = torch.nn.NLLLoss(reduction='none')
        loss = criterion(log_probs, S)
        loss_av = loss.mean()
        return loss, loss_av
    


