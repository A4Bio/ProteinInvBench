from main import Exp
from config import create_parser
import json
import torch
from opencpd.datasets.featurizer import featurize_GTrans
from opencpd.datasets.featurizer import featurize_ProteinMPNN
from opencpd.datasets.featurizer import featurize_GVP

if __name__ == '__main__':
    args = create_parser()
    config = args.__dict__
    # dataset = 'CATH4.2'
    for dataset in ['CATH4.2', 'CATH4.3', 'MPNN']:
        for method in ['PiFold', 'StructGNN', 'GraphTrans', 'GCA', 'AlphaDesign', 'ProteinMPNN', 'GVP']:
            # load config
            params = json.load(open(f"/gaozhangyang/experiments/OpenCPD/results/{dataset}/{method}/model_param.json" , 'r'))
            config.update(params)
            if method=='KWDesign':
                config['recycle_n'] = 3
            
            # load model
            exp = Exp(args)
            exp.method.model.load_state_dict(torch.load(f"/gaozhangyang/experiments/OpenCPD/results/{dataset}/{method}/checkpoint.pth"), strict = False)
            
            with torch.no_grad():
                # sample sequences
                if method in ['PiFold', 'StructGNN', 'GraphTrans', 'GCA', 'AlphaDesign', 'KWDesign']:
                    results = exp.method._save_probs(exp.test_loader.dataset, featurize_GTrans)
                elif method in ['ProteinMPNN']:
                    results = exp.method._save_probs(exp.test_loader.dataset, featurize_ProteinMPNN)
                elif method in ['GVP']:
                    results = exp.method._save_probs(exp.test_loader.dataset, featurize_GVP())
            
            torch.save(results , f"/gaozhangyang/experiments/OpenCPD/results/{dataset}/{method}/results.pt")