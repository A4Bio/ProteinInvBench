import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='/gaozhangyang/experiments/OpenCPD/results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--no_wandb', default=1, type=int)
    
    # CATH
    # dataset parameters
    parser.add_argument('--data_name', default='MPNN', choices=['MPNN', 'PDB', 'CATH4.2', 'TS50', 'CATH4.3'])
    parser.add_argument('--data_root', default='/gaozhangyang/experiments/OpenCPD/data/')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--score_thr', default=70.0, type=float)
    parser.add_argument('--pdb_path', default='/gaozhangyang/drug_dataset/full_pdb/pdb_protein_chains/')
    parser.add_argument('--split_csv', default="/gaozhangyang/experiments/PiFoldV2/data/split/training_samples.csv")
    

    # method parameters
    parser.add_argument('--method', default='GVP', choices=['AlphaDesign', 'PiFold', 'KWDesign', 'GraphTrans', 'StructGNN', 'GVP', 'GCA', 'ProteinMPNN', 'ESMIF'])
    parser.add_argument('--config_file', '-c', default=None, type=str)
    
    # Training parameters
    parser.add_argument('--epoch', default=20, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.00001, type=float, help='Learning rate')
    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--augment_eps', default=0.00, type=float, help='augment_eps')
    parser.add_argument('--removeTS', default=0, type=int, help='remove training and validation samples that have 30+% similarity to TS50 and TS500')

    return parser.parse_args()