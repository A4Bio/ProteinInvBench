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
    # CATH

    # dataset parameters
    parser.add_argument('--data_name', default='CATH', choices=['CATH', 'TS50', 'TS500', 'AlphaFold'])
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    # af dataset parameters
    parser.add_argument('--upid', default='UP000000437_7955_DANRE_v2')
    parser.add_argument('--limit_length', default=0, type=int)
    parser.add_argument('--joint_data', default=0, type=int)
    parser.add_argument('--score_thr', default=70.0, type=float)

    # method parameters
    parser.add_argument('--method', default='PiFold', choices=['PiFold', 'AlphaDesign', 'StructGNN', 'GraphTrans', 'GVP', 'GCA', 'ProteinMPNN', 'ESMIF'])
    parser.add_argument('--config_file', '-c', default=None, type=str)

    # Training parameters
    parser.add_argument('--epoch', default=100, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--augment_eps', default=0.00, type=float, help='augment_eps')
    parser.add_argument('--removeTS', default=0, type=int, help='remove training and validation samples that have 30+% similarity to TS50 and TS500')
    

    # feature parameters
    # parser.add_argument('--full_atom_dis', default=10, type=int)
    parser.add_argument('--node_dist', default=1, type=int)
    parser.add_argument('--node_angle', default=1, type=int)
    parser.add_argument('--node_direct', default=1, type=int)
    parser.add_argument('--edge_dist', default=1, type=int)
    parser.add_argument('--edge_angle', default=1, type=int)
    parser.add_argument('--edge_direct', default=1, type=int)

    parser.add_argument('--edge_dist_remove_Ca', default=0, type=int)
    parser.add_argument('--edge_dist_remove_C', default=0, type=int)
    parser.add_argument('--edge_dist_remove_N', default=0, type=int)
    parser.add_argument('--edge_dist_remove_O', default=0, type=int)
    parser.add_argument('--virtual_num', default=3, type=int)
    parser.add_argument('--edge_dist_only_Ca', default=0, type=int)
    parser.add_argument('--edge_dist_only_Ca_Ca', default=0, type=int)
    parser.add_argument('--edge_dist_only_self', default=0, type=int)

    parser.add_argument('--Ca_Ca', default=1, type=int)
    parser.add_argument('--Ca_C', default=1, type=int)
    parser.add_argument('--Ca_N', default=1, type=int)
    parser.add_argument('--Ca_O', default=1, type=int)
    parser.add_argument('--C_C', default=1, type=int)
    parser.add_argument('--C_N', default=1, type=int)
    parser.add_argument('--C_O', default=1, type=int)
    parser.add_argument('--N_N', default=1, type=int)
    parser.add_argument('--N_O', default=1, type=int)
    parser.add_argument('--O_O', default=1, type=int)


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
    parser.add_argument('--node_context', default=1, type=int)
    parser.add_argument('--edge_context', default=0, type=int)
    parser.add_argument('--AT_layer_num', default=3, type=int)

    # Adesign_plus parameters
    parser.add_argument('--att_layer', default=1, type=int)
    parser.add_argument('--cnn_layer', default=3, type=int)

    # refine
    parser.add_argument('--refine_seq', default=0, type=int)
    parser.add_argument('--refine_itr', default=3, type=int)
    parser.add_argument('--wandb', default=0, type=int)

    parser.add_argument('--egnn', default=0, type=int)

    return parser.parse_args()