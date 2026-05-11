import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from IS2D_models import IS2D_model, model_to_device
from dataset.BioMedicalDataset.SkinSegmentation2018Dataset import MedicalSegmentationDataset
from utils.get_functions import get_device


class BaseSegmentationExperiment(object):
    def __init__(self, args):
        super(BaseSegmentationExperiment, self).__init__()
        self.args = args
        self.args.device = get_device()
        self.use_amp = self.args.device.type == "cuda"

        if self.args.seed_fix:
            self.fix_seed()

        print("STEP1. Load {} Dataset Loaders...".format(self.args.dataset_name))
        self.train_loader, self.val_loader, self.test_loader = self.build_dataloaders()

        print("STEP2. Load MADGNet ...")
        self.model = model_to_device(args, IS2D_model(args))

    def build_dataloaders(self):
        train_dataset = MedicalSegmentationDataset(
            self.args.train_dataset_dir,
            split="train",
            image_size=self.args.image_size,
            augment=True,
            image_dir_name=self.args.image_dir_name,
            mask_dir_name=self.args.mask_dir_name,
        )
        val_dataset = MedicalSegmentationDataset(
            self.args.val_dataset_dir,
            split="val",
            image_size=self.args.image_size,
            augment=False,
            image_dir_name=self.args.image_dir_name,
            mask_dir_name=self.args.mask_dir_name,
        )
        test_dataset = MedicalSegmentationDataset(
            self.args.test_dataset_dir,
            split="test",
            image_size=self.args.image_size,
            augment=False,
            image_dir_name=self.args.image_dir_name,
            mask_dir_name=self.args.mask_dir_name,
        )

        pin_memory = self.args.device.type == "cuda"

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader

    def fix_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

    def move_batch_to_device(self, batch):
        return {
            "image": batch["image"].to(self.args.device, non_blocking=True),
            "region": batch["region"].to(self.args.device, non_blocking=True),
            "distance": batch["distance"].to(self.args.device, non_blocking=True),
            "boundary": batch["boundary"].to(self.args.device, non_blocking=True),
            "name": batch["name"],
        }
