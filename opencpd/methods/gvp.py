from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from opencpd.models import GVP_Model
from .base_method import Base_method


class GVP(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}

    def _build_model(self):
        return GVP_Model(self.args).to(self.device)

    def _cal_recovery(self, dataset, featurizer):
        recovery = []
        subcat_recovery = {}

        for protein in tqdm(dataset):
            p_category = protein['category'] if 'category' in protein.keys() else 'Unknown'
            if p_category not in subcat_recovery.keys():
                subcat_recovery[p_category] = []

            protein = featurizer.collate([protein])
            protein = protein.to(self.device)
            sample = self.model.test_recovery(protein)
            cmp = sample.eq(protein.seq)

            # # TODO: this may ignore score in AF2 DB, lead to higher recovery
            # if 'score' in protein.keys():
            #     score = torch.tensor(protein['score']).cuda()
            # else:
            #     score = torch.ones_like(cmp).cuda()*1000
            # cmp = cmp.view(-1)[score >= self.args.score_thr]
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
            batch = batch.to(self.device)
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)
            logits = self.model(h_V, batch.edge_index, h_E, seq=batch.seq)
            logits, seq = logits[batch.mask], batch.seq[batch.mask]
            loss = self.criterion(logits, seq)
            loss.backward()
        
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()

            mask = batch.mask
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
                batch = batch.to(self.device)
                h_V = (batch.node_s, batch.node_v)
                h_E = (batch.edge_s, batch.edge_v)
                logits = self.model(h_V, batch.edge_index, h_E, seq=batch.seq)
                logits, seq = logits[batch.mask], batch.seq[batch.mask]
                loss = self.criterion(logits, seq)
                mask = batch.mask

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
                batch = batch.to(self.device)
                h_V = (batch.node_s, batch.node_v)
                h_E = (batch.edge_s, batch.edge_v)
                logits = self.model(h_V, batch.edge_index, h_E, seq=batch.seq)
                logits, seq = logits[batch.mask], batch.seq[batch.mask]
                loss = self.criterion(logits, seq)
                mask = batch.mask
                
                test_sum += torch.sum(loss * mask).cpu().data.numpy()
                test_weights += torch.sum(mask).cpu().data.numpy()

                test_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))

            test_recovery, test_subcat_recovery = self._cal_recovery(test_loader.dataset, test_loader.featurizer)
            
        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)
        return test_perplexity, test_recovery, test_subcat_recovery