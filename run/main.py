import logging
import sys
import json
import torch
import os.path as osp
from config import create_parser

import warnings
warnings.filterwarnings('ignore')
from opencpd.utils.recorder import Recorder
from opencpd.utils.main_utils import print_log, output_namespace, set_seed, check_dir, load_config, get_dataset
import wandb
from datetime import datetime
from opencpd.methods import method_maps
import os


class Exp:
    def __init__(self, args, show_params=True):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        if show_params:
            print_log(output_namespace(self.args))
    
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU:',device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _preparation(self):
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        
        # build the method
        if self.args.method!="KWDesign":
            self._build_method()

    def _build_method(self):
        steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, steps_per_epoch)

    def _get_data(self): 
        self.train_loader, self.valid_loader, self.test_loader = get_dataset(self.config)

    
    def _save(self, name=''):
        if self.args.method=='KWDesign':
            torch.save({key:val for key,val in self.method.model.state_dict().items() if "GNNTuning" in key}, osp.join(self.checkpoints_path, name + '.pth'))
        else:
            torch.save(self.method.model.state_dict(), osp.join(self.checkpoints_path, name + '.pth'))

    def _load(self, epoch):
        self.method.model.load_state_dict(torch.load(osp.join(self.checkpoints_path, str(epoch) + '.pth')), strict=False)
    
    def train_KWDesign(self):
        self.args.patience = 3
        self.args.epoch = 5
        recycle_n = self.args.recycle_n
        for cycle in range(1, recycle_n+1):
            self.args.recycle_n = cycle
            current_pth = osp.join(self.args.res_dir, self.args.ex_name, "checkpoints", f"msa{self.args.msa_n}_recycle{self.args.recycle_n}_epoch{self.args.load_epoch}.pth")
            if os.path.exists(current_pth):
                continue
            else:
                self._build_method()
                self.train()
    
    def train(self):
        recorder = Recorder(self.args.patience, verbose=True)
        for epoch in range(self.args.epoch):
            train_loss, train_perplexity = self.method.train_one_epoch(self.train_loader)

            if epoch % self.args.log_step == 0:
                with torch.no_grad():
                    valid_loss, valid_perplexity = self.valid()
                    if self.args.method=='KWDesign':
                        self._save(name=f"msa{self.args.msa_n}_recycle{self.args.recycle_n}_epoch{epoch}")
                        if not os.path.exists(self.args.memory_path):
                            torch.save({"memo_pifold":self.method.model.memo_pifold.memory, "memo_esmif":self.method.model.memo_esmif.memory} , self.args.memory_path)
                    else:
                        self._save(name=str(epoch))
                    # self.test()
                
                print_log('Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Train Perp: {3:.4f} Valid Loss: {4:.4f} Valid Perp: {5:.4f}\n'.format(epoch + 1, len(self.train_loader), train_loss, train_perplexity, valid_loss, valid_perplexity))
                
                if not self.args.no_wandb:
                    wandb.log({"valid_perplexity": valid_perplexity})

                if self.args.method=='KWDesign':
                    recorder(valid_loss, {key:val for key,val in self.method.model.state_dict().items() if "GNNTuning" in key}, self.path)
                else:
                    recorder(valid_loss, self.method.model.state_dict(), self.path)
                    
                if recorder.early_stop:
                    print("Early stopping")
                    logging.info("Early stopping")
                    break
            
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self.method.model.load_state_dict(torch.load(best_model_path), strict=False)

    def valid(self):
        valid_loss, valid_perplexity = self.method.valid_one_epoch(self.valid_loader)

        print_log('Valid Perp: {0:.4f}'.format(valid_perplexity))
        
        return valid_loss, valid_perplexity

    def test(self):
        test_perplexity, test_recovery = self.method.test_one_epoch(self.test_loader)
        print_log('Test Perp: {0:.4f}, Test Rec: {1:.4f}\n'.format(test_perplexity, test_recovery))
        if not self.args.no_wandb:
            wandb.log({"test_perplexity": test_perplexity,
                       "test_acc": test_recovery})

        return test_perplexity, test_recovery


def main():
    args = create_parser()
    config = args.__dict__

    default_params = load_config(osp.join('./configs', args.method + '.py' if args.config_file is None else args.config_file))
    config.update(default_params)
    # args.no_wandb = 1

    if not args.no_wandb:
        os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0"
        wandb.init(project="opencpd", entity="gaozhangyang", config=config, name=args.ex_name)
        
    print(config)
    exp = Exp(args)
    
    # best_model_path = osp.join(exp.path, 'checkpoints/19.pth')
    # exp.method.model.load_state_dict(torch.load(best_model_path))

    print('>>>>>>>>>>>>>>>>>>>>>>>>>> training <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    if args.method == 'KWDesign':
       exp.train_KWDesign() 
    else:
        exp.train()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    test_perp, test_rec = exp.test()
    if not args.no_wandb:
        wandb.log({"test_rec": test_rec.item()})

if __name__ == '__main__':
    main()