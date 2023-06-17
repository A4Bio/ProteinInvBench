import os
import json
import numpy as np
from tqdm import tqdm

import torch.utils.data as data
from .utils import cached_property
from transformers import AutoTokenizer

class ProthermDataset(data.Dataset):
    def __init__(self, path='./',  mode='train', max_length=500, test_name='All', data = None, removeTS=0, version=4.2):
        self.version = version
        self.path = path
        self.mode = mode
        self.max_length = max_length
        self.test_name = test_name
        self.removeTS = removeTS
        if self.removeTS:
            self.remove = json.load(open(self.path+'/remove.json', 'r'))['remove']
        
        if data is None:
            self.data = self.cache_data[mode]
        else:
            self.data = data
        
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/gaozhangyang/model_zoom/transformers")
    
    @cached_property
    def cache_data(self):
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])
        if not os.path.exists(self.path):
            raise "no such file:{} !!!".format(self.path)
        else:
            with open(self.path+'/data_s350.jsonl') as f:
                lines = f.readlines()
            data_valid = []
            for line in tqdm(lines):
                entry = json.loads(line)
                if self.removeTS and entry['name'] in self.remove:
                    continue
                seq = entry['seq']

                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)
                
                bad_chars = set([s for s in seq]).difference(alphabet_set)

                if len(bad_chars) == 0:
                    if len(entry['seq']) <= self.max_length: 
                        chain_length = len(entry['seq'])
                        chain_mask = np.ones(chain_length)
                        data_valid.append({
                            'title':entry['name'],
                            'seq':entry['seq'],
                            'CA':entry['coords']['CA'],
                            'C':entry['coords']['C'],
                            'O':entry['coords']['O'],
                            'N':entry['coords']['N'],
                            'chain_mask': chain_mask,
                            'chain_encoding': 1*chain_mask
                        })
                        
            with open(self.path+'/data_s768.jsonl') as f:
                lines = f.readlines()
            data_train = []
            for line in tqdm(lines):
                entry = json.loads(line)
                if self.removeTS and entry['name'] in self.remove:
                    continue
                seq = entry['seq']

                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)
                
                bad_chars = set([s for s in seq]).difference(alphabet_set)

                if len(bad_chars) == 0:
                    if len(entry['seq']) <= self.max_length: 
                        chain_length = len(entry['seq'])
                        chain_mask = np.ones(chain_length)
                        data_train.append({
                            'title':entry['name'],
                            'seq':entry['seq'],
                            'CA':entry['coords']['CA'],
                            'C':entry['coords']['C'],
                            'O':entry['coords']['O'],
                            'N':entry['coords']['N'],
                            'chain_mask': chain_mask,
                            'chain_encoding': 1*chain_mask
                        })
                        
            with open(self.path+'/data_sym.jsonl') as f:
                lines = f.readlines()
            data_test = []
            for line in tqdm(lines):
                entry = json.loads(line)
                if self.removeTS and entry['name'] in self.remove:
                    continue
                seq = entry['seq']

                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)
                
                bad_chars = set([s for s in seq]).difference(alphabet_set)

                if len(bad_chars) == 0:
                    if len(entry['seq']) <= self.max_length: 
                        chain_length = len(entry['seq'])
                        chain_mask = np.ones(chain_length)
                        data_test.append({
                            'title':entry['name'],
                            'seq':entry['seq'],
                            'CA':entry['coords']['CA'],
                            'C':entry['coords']['C'],
                            'O':entry['coords']['O'],
                            'N':entry['coords']['N'],
                            'chain_mask': chain_mask,
                            'chain_encoding': 1*chain_mask
                        })

            data_dict = {'train':[],'valid':[],'test':[]}
            for data in data_train:
                    data_dict['train'].append(data)
                
            for data in data_test:
                    data_dict['valid'].append(data)
                    
            for data in data_valid:    
                data_dict['test'].append(data)
                    
            return data_dict

    def change_mode(self, mode):
        self.data = self.cache_data[mode]
    
    def __len__(self):
        return len(self.data)
    
    def get_item(self, index):
        return self.data[index]
    
    def __getitem__(self, index):
        return self.data[index]
    
if __name__ == "__main__":
    dataset = ProthermDataset('/usr/data/zyj/OpenCPD/opencpd/datasets/data')
    for data in dataset:
        print(data)   