import csv
import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss, SegmentationMetricTracker


def _create_loaders(args):
    if args.dataset == "Synapse":
        from datasets.dataset_synapse import RandomGenerator, Synapse_dataset

        train_dataset = Synapse_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split="train",
            transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]),
        )
        val_dataset = Synapse_dataset(
            base_dir=args.root_path,
            list_dir=args.list_dir,
            split="val",
            transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]),
        )
    else:
        from datasets.dataset_mydataset import MyDataset, RandomGenerator, ValGenerator

        train_dataset = MyDataset(
            base_dir=args.root_path,
            split="train",
            transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]),
        )
        val_dataset = MyDataset(
            base_dir=args.root_path,
            split="val",
            transform=transforms.Compose([ValGenerator(output_size=[args.img_size, args.img_size])]),
        )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    batch_size = args.batch_size * args.n_gpu
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def _save_best_or_last(model, snapshot_path, batch_loss, best_loss):
    if batch_loss < best_loss:
        save_mode_path = os.path.join(snapshot_path, "best_model.pth")
        torch.save(model.state_dict(), save_mode_path)
        return batch_loss, save_mode_path

    save_mode_path = os.path.join(snapshot_path, "last_model.pth")
    torch.save(model.state_dict(), save_mode_path)
    return best_loss, save_mode_path


def _append_history(history_path, row):
    fieldnames = ["epoch", "learning_rate", "train_loss", "val_loss", "DSC", "mIoU"]
    file_exists = os.path.exists(history_path)
    with open(history_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    db_train, _, train_loader, val_loader = _create_loaders(args)
    print("The length of train set is: {}".format(len(db_train)))
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    history_path = os.path.join(snapshot_path, "history.csv")
    if os.path.exists(history_path):
        os.remove(history_path)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        train_metrics = SegmentationMetricTracker(num_classes)
        lr_ = optimizer.param_groups[0]["lr"]
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", total=len(train_loader),
                                           leave=False):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            train_metrics.update(outputs.detach(), label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # logging.info('Train: iteration : %d/%d, lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (
            #     iter_num, epoch_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            if iter_num % 20 == 0:
                sample_index = 1 if image_batch.size(0) > 1 else 0
                image = image_batch[sample_index, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[sample_index, ...] * 50, iter_num)
                labs = label_batch[sample_index, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        train_dsc, train_miou, _, _ = train_metrics.compute()
        writer.add_scalar('epoch/train_loss', batch_loss, epoch_num + 1)
        writer.add_scalar('epoch/train_dsc', train_dsc, epoch_num + 1)
        writer.add_scalar('epoch/train_miou', train_miou, epoch_num + 1)
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f, DSC: %f, mIoU: %f' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss, train_dsc, train_miou))
        history_row = {
            "epoch": epoch_num + 1,
            "learning_rate": lr_,
            "train_loss": batch_loss,
            "val_loss": "",
            "DSC": "",
            "mIoU": "",
        }
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0
            val_metrics = SegmentationMetricTracker(num_classes)
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                                   total=len(val_loader), leave=False):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    val_metrics.update(outputs, label_batch)
                    batch_dice_loss += loss_dice.item()
                    batch_ce_loss += loss_ce.item()

                batch_ce_loss /= len(val_loader)
                batch_dice_loss /= len(val_loader)
                batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
                val_dsc, val_miou, _, _ = val_metrics.compute()
                writer.add_scalar('epoch/val_loss', batch_loss, epoch_num + 1)
                writer.add_scalar('epoch/val_dsc', val_dsc, epoch_num + 1)
                writer.add_scalar('epoch/val_miou', val_miou, epoch_num + 1)
                logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f, DSC: %f, mIoU: %f' % (
                    epoch_num, batch_loss, batch_ce_loss, batch_dice_loss, val_dsc, val_miou))
                history_row["val_loss"] = batch_loss
                history_row["DSC"] = val_dsc
                history_row["mIoU"] = val_miou
                best_loss, save_mode_path = _save_best_or_last(model, snapshot_path, batch_loss, best_loss)
                logging.info("save model to {}".format(save_mode_path))
        _append_history(history_path, history_row)

    writer.close()
    return "Training Finished!"


def trainer_mydataset(args, model, snapshot_path):
    return trainer_synapse(args, model, snapshot_path)
