import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--no_wandb', default=1, type=int)
    
    # CATH
    # dataset parameters
    parser.add_argument('--data_name', default='CATH', choices=['MPNN', 'PDB', 'CATH', 'TS50', 'CATH4.3'])
    parser.add_argument('--data_root', default='/gaozhangyang/experiments/OpenCPD/data/')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--score_thr', default=70.0, type=float)
    parser.add_argument('--pdb_path', default='/gaozhangyang/drug_dataset/full_pdb/pdb_protein_chains/')
    parser.add_argument('--split_csv', default="/gaozhangyang/experiments/PiFoldV2/data/split/training_samples.csv")
    

    # method parameters
    parser.add_argument('--method', default='KWDesign', choices=['PiFold', 'PiFoldV2', 'KWDesign'])
    parser.add_argument('--config_file', '-c', default=None, type=str)
    parser.add_argument('--fix_gnn', default=1, type=int)
    parser.add_argument('--fix_esm', default=1, type=int)
    parser.add_argument('--recycle_n', default=1, type=int)
    parser.add_argument('--msa_n', default=2, type=int)
    parser.add_argument('--temporature', default=0.05, type=float)
    parser.add_argument('--tunning_layers_n', default=4, type=int)
    parser.add_argument('--tunning_layers_dim', default=256, type=int)
    parser.add_argument('--tunning_dropout', default=0.1, type=float)
    parser.add_argument('--input_design_dim', default=128, type=int)
    parser.add_argument('--input_esm_dim', default=1280, type=int)
    parser.add_argument("--use_LM", default=1, type=int)
    parser.add_argument("--use_gearnet", default=0, type=int)
    parser.add_argument("--use_esmif", default=1, type=int)
    parser.add_argument("--load_memory", default=1, type=int)
    parser.add_argument("--use_confembed", default=1, type=int)
    parser.add_argument("--memory_path", default="/gaozhangyang/experiments/PiFoldV2/results/memotuning_use_LM_esmif/checkpoints/memory.pth") # /gaozhangyang/experiments/PiFoldV2/results/memotuning_use_LM_esmif/checkpoints/memory.pth
    
    parser.add_argument("--load_epoch", default=3, type=int)
    
    

    # Training parameters
    parser.add_argument('--epoch', default=5, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--augment_eps', default=0.00, type=float, help='augment_eps')
    parser.add_argument('--removeTS', default=0, type=int, help='remove training and validation samples that have 30+% similarity to TS50 and TS500')
    

    # feature parameters
    # parser.add_argument('--full_atom_dis', default=10, type=int)
    parser.add_argument('--use_gvp_feat', default=0, type=int) # for rebuttal

    parser.add_argument('--updating_edges', default=4, type=int)
    parser.add_argument('--node_dist', default=1, type=int)
    parser.add_argument('--node_angle', default=1, type=int)
    parser.add_argument('--node_direct', default=1, type=int)
    parser.add_argument('--edge_dist', default=1, type=int)
    parser.add_argument('--edge_angle', default=1, type=int)
    parser.add_argument('--edge_direct', default=1, type=int)


    # debug parameters
    parser.add_argument('--proteinmpnn_type', default=0, type=int)
    parser.add_argument('--num_decoder_layers1', default=3, type=int)
    parser.add_argument('--kernel_size1', default=3, type=int)
    parser.add_argument('--act_type', default='silu', type=str)
    parser.add_argument('--num_decoder_layers2', default=3, type=int)
    parser.add_argument('--kernel_size2', default=3, type=int)
    parser.add_argument('--glu', default=0, type=int)
    parser.add_argument('--dihedral_type', default=0, type=int)
    parser.add_argument('--NAT', default=3, type=int)

    # AlphaDesign parameters
    parser.add_argument('--num_encoder_layers', default=10, type=int)
    parser.add_argument('--node_net', default='AttMLP', choices=['AttMLP', 'GCN', 'GAT', 'QKV'])
    parser.add_argument('--edge_net', default='EdgeMLP', choices=['None','EdgeMLP', 'DualEGraph'])
    parser.add_argument('--node_context', default=1, type=int)
    parser.add_argument('--edge_context', default=0, type=int)
    parser.add_argument('--autoregressive', default=0, type=int)
    parser.add_argument('--AT_layer_num', default=3, type=int)

    # Adesign_plus parameters
    parser.add_argument('--att_layer', default=1, type=int)
    parser.add_argument('--cnn_layer', default=3, type=int)

    return parser.parse_args()