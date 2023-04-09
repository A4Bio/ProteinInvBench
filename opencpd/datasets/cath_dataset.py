import os
import json
import numpy as np
from tqdm import tqdm

import torch.utils.data as data
from opencpd.datasets.utils import cached_property


class CATHDataset(data.Dataset):
    def __init__(self, path='./',  mode='train', max_length=500, test_name='All', data = None, removeTS=0):
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
    
    @cached_property
    def cache_data(self):
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])
        if not os.path.exists(self.path):
            raise "no such file:{} !!!".format(self.path)
        else:
            with open(self.path+'/chain_set.jsonl') as f:
                lines = f.readlines()
            data_list = []
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
                        data_list.append({
                            'title':entry['name'],
                            'seq':entry['seq'],
                            'CA':entry['coords']['CA'],
                            'C':entry['coords']['C'],
                            'O':entry['coords']['O'],
                            'N':entry['coords']['N']
                        })
            
            with open(self.path+'/chain_set_splits.json') as f:
                dataset_splits = json.load(f)
            
            if self.test_name == 'L100':
                with open(self.path+'/test_split_L100.json') as f:
                    test_splits = json.load(f)
                dataset_splits['test'] = test_splits['test']

            if self.test_name == 'sc':
                with open(self.path+'/test_split_sc.json') as f:
                    test_splits = json.load(f)
                dataset_splits['test'] = test_splits['test']
            
            name2set = {}
            name2set.update({name:'train' for name in dataset_splits['train']})
            name2set.update({name:'valid' for name in dataset_splits['validation']})
            name2set.update({name:'test' for name in dataset_splits['test']})

            data_dict = {'train':[], 'valid':[], 'test':[]}
            for data in data_list:
                if name2set.get(data['title']):
                    if name2set[data['title']] == 'train':
                        data_dict['train'].append(data)           
                    if name2set[data['title']] == 'valid':
                        data_dict['valid'].append(data)
                    if name2set[data['title']] == 'test':
                        data['category'] = 'Unkown'
                        data['score'] = 100.0
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