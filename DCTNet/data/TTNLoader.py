#! /usr/bin/python
# -*- encoding: utf-8 -*-
"""
@author LeslieZhao
@date 20220721
"""
import os
from torchvision import transforms
import PIL.Image as Image
from data.DataLoader import DatasetBase
import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TTNData(DatasetBase):
    def __init__(self, slice_id=0, slice_count=1, dist=False, **kwargs):
        super().__init__(slice_id, slice_count, dist, **kwargs)

        # self.transform = transforms.Compose([
        #     transforms.Resize([256,256]),
        #     transforms.RandomResizedCrop(256,scale=(0.8,1.2)),
        #     transforms.RandomRotation(degrees=(-90,90)),
        #      transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        self.transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=256),
                A.ShiftScaleRotate(
                    shift_limit=0.2, scale_limit=[-0.6, 0.2], rotate_limit=30, p=0.5
                ),
                #                 A.RandomBrightnessContrast(brightness_limit=[-0, 0.2], contrast_limit=[0.5, 1], p=0.5),
                A.Perspective(scale=(0.09, 0.1), p=0.5),
                A.CenterCrop(512, 512),
                A.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                    always_apply=True,
                ),
                ToTensorV2(always_apply=True),
            ],
            additional_targets={"target": "image"},
        )

        if kwargs["eval"]:
            # self.transform = transforms.Compose([
            # transforms.Resize([256,256]),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.transform = A.Compose(
                [
                    A.SmallestMaxSize(max_size=256),
                    A.Normalize(
                        mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5),
                        always_apply=True,
                    ),
                    ToTensorV2(always_apply=True),
                ],
                additional_targets={"target": "image"},
            )

            self.length = 100

        src_root = kwargs["src_root"]
        tgt_root = kwargs["tgt_root"]

        # transformed = self.transform(image=image_A, target=image_B)
        #             image_A = transformed["image"]
        #             image_B = transformed["target"]

        self.src_paths = [
            os.path.join(src_root, f)
            for f in os.listdir(src_root)
            if f.endswith(".png") or f.endswith(".jpg")
        ]
        self.tgt_paths = [
            os.path.join(tgt_root, f)
            for f in os.listdir(tgt_root)
            if f.endswith(".png") or f.endswith(".jpg")
        ]
        self.src_length = len(self.src_paths)
        self.tgt_length = len(self.tgt_paths)
        random.shuffle(self.src_paths)
        random.shuffle(self.tgt_paths)

        (
            self.mx_left_eye_all,
            self.mn_left_eye_all,
            self.mx_right_eye_all,
            self.mn_right_eye_all,
            self.mx_lip_all,
            self.mn_lip_all,
        ) = np.load(kwargs["score_info"])

    def __getitem__(self, i):
        src_idx = i % self.src_length
        tgt_idx = i % self.tgt_length

        src_path = self.src_paths[src_idx]
        tgt_path = self.tgt_paths[tgt_idx]
        exp_path = src_path.replace("img", "express")[:-3] + "npy"

        image_src = np.array(Image.open(src_path))
        image_tgt = np.array(Image.open(tgt_path))

        # with Image.open(src_path) as img:
        #     srcImg = self.transform(img)

        # with Image.open(tgt_path) as img:
        #     tgtImg = self.transform(img)

        image_src = self.transform(image=image_src)["image"]
        image_tgt = self.transform(image=image_tgt)["image"]

        # image_src = transformed["image"]?
        # image_tgt = transformed["target"]

        score = np.load(exp_path)
        score[0] = (score[0] - self.mn_left_eye_all) / (
            self.mx_left_eye_all - self.mn_left_eye_all
        )
        score[1] = (score[1] - self.mn_right_eye_all) / (
            self.mx_right_eye_all - self.mn_right_eye_all
        )
        score[2] = (score[2] - self.mn_lip_all) / (self.mx_lip_all - self.mn_lip_all)
        score = torch.from_numpy(score.astype(np.float32))

        return image_src, image_tgt, score

    def __len__(self):
        # return max(self.src_length,self.tgt_length)
        if hasattr(self, "length"):
            return self.length
        else:
            return 10000
