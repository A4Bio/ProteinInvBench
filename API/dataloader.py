import copy
import os.path as osp

from .cath_dataset import CATH
from .alphafold_dataset import AlphaFold
from .ts_dataset import TS

from .dataloader_gtrans import DataLoader_GTrans
from .featurizer import featurize_GTrans, featurize_AF, featurize_ProteinMPNN, featurize_Inversefolding
from .dataloader_gvp import DataLoader_GVP, featurize_GVP


def load_data(data_name, method, batch_size, data_root, upid, limit_length, joint_data, max_nodes=3000, num_workers=8, removeTS=0, **kwargs):
    if data_name == 'CATH' or data_name == 'TS':
        cath_set = CATH(osp.join(data_root, 'cath'), mode='train', test_name='All', removeTS=removeTS)
        train_set, valid_set, test_set = map(lambda x: copy.copy(x), [cath_set] * 3)
        valid_set.change_mode('valid')
        test_set.change_mode('test')
        if data_name == 'TS':
            test_set = TS(osp.join(data_root, 'ts'))
        collate_fn = featurize_GTrans
    elif data_name == 'AlphaFold':
        af_set = AlphaFold(osp.join(data_root, 'af2db'), upid=upid, mode='train', limit_length=limit_length, joint_data=joint_data)
        train_set, valid_set, test_set = map(lambda x: copy.copy(x), [af_set] * 3)
        valid_set.change_mode('valid')
        test_set.change_mode('test')
        collate_fn = featurize_AF

    if method in ['PiFold', 'AlphaDesign', 'GraphTrans', 'StructGNN', 'GCA', 'Adesign_plus']:
        train_loader = DataLoader_GTrans(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        valid_loader = DataLoader_GTrans(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    elif method == 'GVP':
        featurizer = featurize_GVP()
        train_loader = DataLoader_GVP(train_set, num_workers=num_workers, featurizer=featurizer, max_nodes=max_nodes)
        valid_loader = DataLoader_GVP(valid_set, num_workers=num_workers, featurizer=featurizer, max_nodes=max_nodes)
        test_loader = DataLoader_GVP(test_set, num_workers=num_workers, featurizer=featurizer, max_nodes=max_nodes)
    elif method == 'ProteinMPNN':
        collate_fn = featurize_ProteinMPNN
        train_loader = DataLoader_GTrans(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        valid_loader = DataLoader_GTrans(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    elif method == 'ESMIF':
        collate_fn = featurize_Inversefolding
        train_loader = DataLoader_GTrans(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        valid_loader = DataLoader_GTrans(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader



def make_cath_loader(test_set, method, batch_size, max_nodes=3000, num_workers=8):
    if method in ['PiFold','ADesign', 'GraphTrans', 'StructGNN', 'GCA']:
        collate_fn = featurize_GTrans
        test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    elif method == 'GVP':
        featurizer = featurize_GVP()
        test_loader = DataLoader_GVP(test_set, num_workers=num_workers, featurizer=featurizer, max_nodes=max_nodes)
    elif method == 'ProteinMPNN':
        collate_fn = featurize_ProteinMPNN
        test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    elif method == 'ESMIF':
        collate_fn = featurize_Inversefolding
        test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return test_loader