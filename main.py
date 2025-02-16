import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import torch
import random
import numpy as np
from trainer import HiEndMAETrainer

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="hi)hi_end_mae_vit_large_12p", choices=["hi_end_mae_vit_large_12p", "hi_end_mae_vit_base_16p", "hi_end_mae_vit_large_16p"], help='model')
parser.add_argument('--base_dir1', type=str, default="", help='dir')
parser.add_argument('--base_lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--epochs', type=int, default=1000, help='train epoch')
parser.add_argument('--img_size', type=int, default=96, help='img size of per batch')
parser.add_argument('--patch_size', type=int, default=12, help='patch size of per img')
parser.add_argument('--in_chans', type=int, default=1, help='input channels')
parser.add_argument('--pos_embed_type', type=str, default="sincos")
parser.add_argument('--seed', type=int, default=41, help='random seed')
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--ckpt_dir', type=str, default="./ckpt/hi_end_mae_vit_large_12p")
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.95)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--wde', type=float, default=0.2)
parser.add_argument('--wp_ep', type=int, default=5)
parser.add_argument('--mask_ratio', type=float, default=0.75)
args = parser.parse_args()

def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    args = parser.parse_args()

    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    seed_torch(args.seed)
    trainer = DenseMAETrainer(args=args)

    trainer.run()
    