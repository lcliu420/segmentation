import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_mydataset import MyDataset
from datasets.dataset_synapse import Synapse_dataset
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import calculate_metric_percase, test_single_volume


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=None, help="root dir for dataset")
    parser.add_argument("--dataset", type=str, default="WL", choices=["Synapse", "WL", "NBI"],
                        help="dataset name")
    parser.add_argument("--num_classes", type=int, default=None, help="output channels of network")
    parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir")
    parser.add_argument("--max_iterations", type=int, default=30000, help="maximum iteration number to train")
    parser.add_argument("--max_epochs", type=int, default=150, help="maximum epoch number to train")
    parser.add_argument("--batch_size", type=int, default=24, help="batch size per gpu")
    parser.add_argument("--img_size", type=int, default=224, help="input patch size")
    parser.add_argument("--is_savenii", action="store_true", help="whether to save results during inference")
    parser.add_argument("--test_save_dir", type=str, default=None, help="saving prediction dir")
    parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")
    parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation learning rate")
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
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--split_name", default="test", choices=["train", "val", "test"], help="dataset split")
    return parser.parse_args()


def resolve_dataset_settings(args):
    if args.dataset == "Synapse":
        args.root_path = args.root_path or "./datasets/Synapse"
        args.volume_path = os.path.join(args.root_path, "test_vol_h5")
        args.list_dir = args.list_dir or "./lists/lists_Synapse"
        args.num_classes = args.num_classes or args.n_class or 9
        args.z_spacing = 1
    else:
        args.root_path = args.root_path or os.path.join("datasets", "MyDataset", args.dataset)
        args.volume_path = None
        args.list_dir = None
        args.num_classes = args.num_classes or args.n_class or 2
        args.z_spacing = None


def prepare_2d_input(image, img_size):
    if image.ndim == 2:
        image = image[..., None]
    original_h, original_w = image.shape[:2]
    if original_h != img_size or original_w != img_size:
        image = zoom(image, (img_size / original_h, img_size / original_w, 1), order=3)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return image


def save_prediction_mask(prediction, save_path, num_classes):
    if num_classes == 2:
        prediction = (prediction > 0).astype(np.uint8) * 255
    else:
        prediction = prediction.astype(np.uint8)
    Image.fromarray(prediction).save(save_path)


def inference_synapse(args, model, test_save_path=None):
    db_test = Synapse_dataset(base_dir=args.volume_path, split=args.split_name, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("%d test iterations per epoch", len(testloader))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"][0]
        metric_i = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
        )
        metric_list += np.array(metric_i)
        logging.info("idx %d case %s mean_dice %f mean_hd95 %f",
                     i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info("Mean class %d mean_dice %f mean_hd95 %f",
                     i, metric_list[i - 1][0], metric_list[i - 1][1])
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info("Testing performance in best val model: mean_dice : %f mean_hd95 : %f", performance, mean_hd95)
    return "Testing Finished!"


def inference_mydataset(args, model, test_save_path=None):
    db_test = MyDataset(base_dir=args.root_path, split=args.split_name, transform=None)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("%d test iterations per epoch", len(testloader))
    model.eval()
    metric_list = np.zeros((args.num_classes - 1, 2), dtype=np.float64)

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        image = sampled_batch["image"][0].cpu().numpy()
        label = sampled_batch["label"][0].cpu().numpy()
        case_name = sampled_batch["case_name"][0]

        input_tensor = prepare_2d_input(image, args.img_size).cuda()
        with torch.no_grad():
            outputs = model(input_tensor)
            prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()

        if prediction.shape != label.shape:
            prediction = zoom(
                prediction,
                (label.shape[0] / prediction.shape[0], label.shape[1] / prediction.shape[1]),
                order=0,
            )

        metric_i = []
        for class_index in range(1, args.num_classes):
            metric_i.append(calculate_metric_percase(prediction == class_index, label == class_index))

        metric_list += np.array(metric_i)
        logging.info("idx %d case %s mean_dice %f mean_hd95 %f",
                     i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])

        if test_save_path is not None:
            save_prediction_mask(prediction, os.path.join(test_save_path, f"{case_name}_pred.png"), args.num_classes)

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info("Mean class %d mean_dice %f mean_hd95 %f",
                     i, metric_list[i - 1][0], metric_list[i - 1][1])
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info("Testing performance in best val model: mean_dice : %f mean_hd95 : %f", performance, mean_hd95)
    return "Testing Finished!"


def main():
    args = parse_args()
    resolve_dataset_settings(args)
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

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    snapshot = os.path.join(args.output_dir, "best_model.pth")
    if not os.path.exists(snapshot):
        snapshot = os.path.join(args.output_dir, "last_model.pth")
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet", msg)
    snapshot_name = os.path.basename(snapshot)

    log_folder = "./test_log"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_folder, snapshot_name + ".txt"), level=logging.INFO,
                        format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = args.test_save_dir or os.path.join(args.output_dir, "predictions")
        os.makedirs(args.test_save_dir, exist_ok=True)
        test_save_path = args.test_save_dir
    else:
        test_save_path = None

    if args.dataset == "Synapse":
        inference_synapse(args, net, test_save_path)
    else:
        inference_mydataset(args, net, test_save_path)


if __name__ == "__main__":
    main()
