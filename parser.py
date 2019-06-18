import argparse
parser = argparse.ArgumentParser(description='PyTorch Reimplementation of Pose\
Encoding for Robust Skeleton-Based Action Recognition')

# ========================= Data Preprocess  ==========================
parser.add_argument('--normalization', type=str, choices=['default', 'my', 'none'],
 default='default')
parser.add_argument('--ftrans', dest='ftrans', action='store_true')
parser.add_argument('--strans', dest='ftrans', action='store_false')
parser.set_defaults(ftrans=True)
parser.add_argument('--modality',type=str, choices=['', '_hidden', '_recovered'])
parser.set_defaults(modality='_recovered')

# ========================= Learning Configs ==========================
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=140,
                    help='upper epoch limit')     
parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')  
parser.add_argument('--lr_decay', dest='lr_decay', action='store_true')
parser.add_argument('--no-lr_decay', dest='lr_decay', action='store_false')
parser.set_defaults(lr_decay=False)   

# ========================= Model Configs ============================
parser.add_argument('--hidden_unit', type=int, default=30,
                    help='hidden units')
parser.add_argument('--num_layers', type=int, default=4,
                    help='num_layers')  
parser.add_argument('--mean_after_fc', dest='mean_after_fc',
 action='store_true')
parser.add_argument('--no-mean_after_fc', dest='mean_after_fc', 
 action='store_false')
parser.set_defaults(mean_after_fc=True)
parser.add_argument('--mask_empty_frame', dest='mask_empty_frame', action='store_true')
parser.add_argument('--no-mask_empty_frame', dest='mask_empty_frame', action='store_false')
parser.set_defaults(mask_frame=False)

# ========================= Runtime Configs ==========================
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--num_workers', type = int, default=4)
parser.add_argument('--gpus', nargs='+', type=str, default=None)
parser.add_argument('--dataset_dir', default='./dataset/', 
    help="root directory for all the datasets")
parser.add_argument('--checkpoint_folder', type=str, default='./checkpoints/')
parser.add_argument('--figure_folder',type=str, default='./figures/')
parser.add_argument('--resume', action='store_true',
                    help='test model of version')
parser.add_argument('--version', type=str, default='')
