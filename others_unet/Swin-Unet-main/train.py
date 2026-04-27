import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_mydataset, trainer_synapse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=None, help="root dir for dataset")
    parser.add_argument("--dataset", type=str, default="WL", choices=["Synapse", "WL", "NBI"],
                        help="dataset name")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir for Synapse")
    parser.add_argument("--num_classes", type=int, default=None, help="output channels of network")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir")
    parser.add_argument("--max_iterations", type=int, default=30000, help="maximum iteration number to train")
    parser.add_argument("--max_epochs", type=int, default=150, help="maximum epoch number to train")
    parser.add_argument("--batch_size", type=int, default=24, help="batch size per gpu")
    parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
    parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")
    parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="input patch size")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", default=None, nargs="+", help="Modify config options by adding 'KEY VALUE' pairs.")
    parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
    parser.add_argument("--cache-mode", type=str, default="part", choices=["no", "full", "part"],
                        help="dataset cache mode")
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
    parser.add_argument("--use-checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--amp-opt-level", type=str, default="O1", choices=["O0", "O1", "O2"],
                        help="mixed precision opt level")
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--throughput", action="store_true", help="Test throughput only")
    parser.add_argument("--n_class", default=None, type=int, help="legacy alias for num_classes")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--eval_interval", default=1, type=int)
    return parser.parse_args()


def resolve_dataset_settings(args):
    if args.dataset == "Synapse":
        args.root_path = args.root_path or "./datasets/Synapse"
        args.root_path = os.path.join(args.root_path, "train_npz")
        args.list_dir = args.list_dir or "./lists/lists_Synapse"
        args.num_classes = args.num_classes or args.n_class or 9
        trainer = trainer_synapse
    else:
        args.root_path = args.root_path or os.path.join("datasets", "MyDataset", args.dataset)
        args.list_dir = None
        args.num_classes = args.num_classes or args.n_class or 2
        trainer = trainer_mydataset
    return trainer


def main():
    args = parse_args()
    trainer = resolve_dataset_settings(args)
    config = get_config(args)

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    os.makedirs(args.output_dir, exist_ok=True)

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_from(config)
    trainer(args, net, args.output_dir)


if __name__ == "__main__":
    main()
