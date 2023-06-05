from main import Exp
from config import create_parser
import json
import torch
from opencpd.datasets.featurizer import featurize_GTrans
from opencpd.datasets.featurizer import featurize_ProteinMPNN
from opencpd.datasets.featurizer import featurize_GVP
import os.path as osp

if __name__ == '__main__':
    args = create_parser()
    config = args.__dict__
    with open(osp.join('/gaozhangyang/experiments/OpenCPD/data/casp15','casp15.jsonl')) as f:
        lines = f.readlines()
    
    classification = {}
    for line in lines:
        entry = json.loads(line)
        classification[entry['name']] = entry['classification']
                
    for dataset in [ 'MPNN']: # CATH4.2', 'CATH4.3',
        for method in ['PiFold']: #, 'StructGNN', 'GraphTrans', 'AlphaDesign', 'ProteinMPNN', 'GVP', 'GCA', 'KWDesign'
            sv_path = f"/gaozhangyang/experiments/OpenCPD/results/{dataset}/{method}"
            # if osp.exists(f"{sv_path}/results_casp15.pt"):
            #     continue
            
            # load config
            params = json.load(open(f"{sv_path}/model_param.json" , 'r'))
            config.update(params)
            
            config['ex_name'] = f"{args.data_name}/{args.method}"
            config['test_casp'] = True
            
            if method=='KWDesign':
                config['recycle_n'] = 3
            
            # load model
            exp = Exp(args)
            
            if method=='KWDesign':
                exp._build_method()
                params = torch.load(osp.join(sv_path, "checkpoints", f"msa{exp.method.args.msa_n}_recycle{exp.method.args.recycle_n}_epoch{exp.method.args.load_epoch}.pth"))
                exp.method.model.load_state_dict(params, strict=False)
            else:
                exp.method.model.load_state_dict(torch.load(f"{sv_path}/checkpoint.pth"), strict = False)
            
            exp.method.model.eval()
            with torch.no_grad():
                # sample sequences
                if method in ['PiFold', 'StructGNN', 'GraphTrans', 'GCA', 'AlphaDesign', 'KWDesign']:
                    results = exp.method._save_probs(exp.test_loader.dataset, featurize_GTrans)
                elif method in ['ProteinMPNN']:
                    results = exp.method._save_probs(exp.test_loader.dataset, featurize_ProteinMPNN)
                elif method in ['GVP']:
                    results = exp.method._save_probs(exp.test_loader.dataset, featurize_GVP())
            
            classification_list = []
            for name in results['title']:
                classification_list.append(classification[name])
            results['classification'] = classification_list
            
            torch.save(results , f"{sv_path}/results_casp15.pt")