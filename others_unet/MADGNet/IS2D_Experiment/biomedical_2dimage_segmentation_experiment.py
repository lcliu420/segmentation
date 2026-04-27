import os

import numpy as np
import pandas as pd
import torch

from ._IS2Dbase import BaseSegmentationExperiment
from utils.calculate_metrics import BMIS_Metrics_Calculator
from utils.load_functions import load_model
from utils.save_functions import save_checkpoint, save_history, save_prediction_masks


class ISICSegmentationExperiment(BaseSegmentationExperiment):
    def __init__(self, args):
        super(ISICSegmentationExperiment, self).__init__(args)
        self.metrics_calculator = BMIS_Metrics_Calculator(args.metric_list)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs,
            eta_min=self.args.min_learning_rate,
        )

        best_dsc = -1.0
        history_rows = []

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_one_epoch(epoch, optimizer)
            scheduler.step()

            val_loss, val_metrics = self.evaluate(self.val_loader, split="val", epoch=epoch)
            lr = optimizer.param_groups[0]["lr"]
            history_rows.append(
                {
                    "epoch": epoch,
                    "learning_rate": lr,
                    "train_loss": np.round(train_loss, 6),
                    "val_loss": np.round(val_loss, 6),
                    **{metric: val_metrics[metric] for metric in self.args.metric_list},
                }
            )
            save_history(self.args, pd.DataFrame(history_rows))

            is_best = val_metrics["DSC"] >= best_dsc
            if is_best:
                best_dsc = val_metrics["DSC"]

            print(
                "EPOCH {} | train_loss {:.4f} | val_loss {:.4f} | val_DSC {:.4f} | val_mIoU {:.4f}".format(
                    epoch,
                    train_loss,
                    val_loss,
                    val_metrics["DSC"],
                    val_metrics["mIoU"],
                )
            )

            save_checkpoint(
                self.args,
                self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_dsc,
                is_best=is_best,
                is_last=True,
                save_epoch_snapshot=(epoch == self.args.epochs),
            )

        save_history(self.args, pd.DataFrame(history_rows))

    def train_one_epoch(self, epoch, optimizer):
        self.model.train()
        running_loss = 0.0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            batch = self.move_batch_to_device(batch)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(batch["image"], mode="train")
                loss, _ = self.model._calculate_multi_task_criterion(
                    outputs,
                    batch["region"],
                    batch["distance"],
                    batch["boundary"],
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch["image"].size(0)
            total += batch["image"].size(0)

            if batch_idx % self.args.step == 0:
                progress = np.round(batch_idx / len(self.train_loader) * 100, 2)
                print("TRAIN EPOCH {} | {}/{} ({}%)".format(epoch, batch_idx, len(self.train_loader), progress))

        return running_loss / max(total, 1)

    def evaluate(self, loader, split, epoch=None, save_predictions=False):
        self.model.eval()
        total_loss = 0.0
        total = 0
        total_metrics_dict = {metric: [] for metric in self.args.metric_list}
        prediction_items = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                batch = self.move_batch_to_device(batch)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output = self.model(batch["image"], mode="test")
                    loss = self.model._calculate_criterion(output, batch["region"])

                predictions = torch.sigmoid(output)
                for idx in range(predictions.size(0)):
                    metrics_dict = self.metrics_calculator.get_metrics_dict(predictions[idx], batch["region"][idx])
                    for metric in self.args.metric_list:
                        total_metrics_dict[metric].append(metrics_dict[metric])

                    if save_predictions:
                        prediction_items.append((batch["name"][idx], predictions[idx].detach().cpu()))

                total_loss += loss.item() * batch["image"].size(0)
                total += batch["image"].size(0)

                if batch_idx % self.args.step == 0:
                    progress = np.round(batch_idx / len(loader) * 100, 2)
                    print("{} EPOCH {} | {}/{} ({}%)".format(split.upper(), epoch, batch_idx, len(loader), progress))

        averaged_metrics = {
            metric: np.round(np.mean(values), 4) if values else 0.0
            for metric, values in total_metrics_dict.items()
        }

        if save_predictions and prediction_items:
            save_prediction_masks(self.args, prediction_items, split)

        return total_loss / max(total, 1), averaged_metrics

    def inference(self, split="test"):
        print("INFERENCE")
        checkpoint = load_model(self.args, self.model)
        checkpoint_name = os.path.basename(checkpoint["checkpoint_path"])
        loader = self.val_loader if split == "val" else self.test_loader
        _, metrics = self.evaluate(loader, split=split, epoch=checkpoint.get("epoch"), save_predictions=self.args.save_predictions)
        return metrics, checkpoint_name
