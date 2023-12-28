# Copyright (c) CAIRI AI Lab. All rights reserved

import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume_from', type=str, default=None, help='the checkpoint file to resume from')
    parser.add_argument('--auto_resume', action='store_true', default=False,
                        help='When training was interupted, resume from the latest checkpoint')
    parser.add_argument('--test', action='store_true', default=False, help='Only performs testing')

    # dataset parameters
    parser.add_argument('--data_name', '-d', default='CATH', type=str,
                        choices=['CATH', 'TS50', 'TS500', 'AlphaFold'])
    parser.add_argument('--data_root', default='./data/', type=str)
    parser.add_argument('--batch_size', '-b', default=16, type=int, help='Training batch size')
    parser.add_argument('--num_workers', default=4, type=int)
    # af dataset parameters
    parser.add_argument('--upid', default='UP000000437_7955_DANRE_v2')
    parser.add_argument('--limit_length', default=0, type=int)
    parser.add_argument('--joint_data', default=0, type=int)
    parser.add_argument('--score_thr', default=70.0, type=float)

    # method parameters
    parser.add_argument('--method', '-m', default='PiFold', type=str,
                        choices=['PiFold', 'AlphaDesign', 'StructGNN', 'GraphTrans', 'GVP', 'GCA', 'ProteinMPNN', 'ESMIF'],
                        help='Name of CPD method to train (default: "PiFold")')
    parser.add_argument('--config_file', '-c', default=None, type=str,
                        help='Path to the default config file')

    # Training parameters
    parser.add_argument('--epoch', default=100, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int, help='Log interval by step')
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer sgd momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=0., type=float, help='Weight decay')
    parser.add_argument('--sched', default='onecycle', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "onecycle"')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_epoch', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--patience', default=20, type=int,
                        help='patient epochs for early stopping')
    parser.add_argument('--augment_eps', default=0.0, type=float, help='augment_eps')
    parser.add_argument('--removeTS', default=0, type=int,
                        help='remove training and validation samples that have 30+% similarity to TS50 and TS500')

    # feature parameters
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

    return parser
