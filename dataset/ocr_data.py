import warnings

warnings.filterwarnings('ignore')
import os
from torch.utils.data import Dataset
from dataset.transforms import CVColorJitter, CVDeterioration, CVGeometry
from torchvision import transforms
import cv2
import math
import random
from PIL import Image
import numpy as np
import json


class ImageDataset(Dataset):
    def __init__(self,
                 src: str,
                 jsonl: str,
                 is_training: bool = True,
                 img_h: int = 32,
                 img_w: int = 100,
                 data_aug: bool = True,
                 multiscales: bool = True,
                 convert_mode: str = 'RGB',
                 ):
        self.data_aug = data_aug
        self.convert_mode = convert_mode
        self.img_h = img_h
        self.img_w = img_w
        self.multiscales = multiscales
        self.is_training = is_training
        self.src = src
        if not os.path.exists(jsonl):
            self.data = ['{"filename": "./dataset/images/000000.png", "text": "4"}'] * 100
        else:
            self.data = open(jsonl).read().split('\n')[:-1]
        if self.is_training and self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def resize_multiscales(self, img, borderType=cv2.BORDER_CONSTANT):
        def _resize_ratio(img, ratio, fix_h=True):
            if ratio * self.img_w < self.img_h:
                if fix_h:
                    trg_h = self.img_h
                else:
                    trg_h = int(ratio * self.img_w)
                trg_w = self.img_w
            else:
                trg_h, trg_w = self.img_h, int(self.img_h / ratio)
            img = cv2.resize(img, (trg_w, trg_h))
            pad_h, pad_w = (self.img_h - trg_h) / 2, (self.img_w - trg_w) / 2
            top, bottom = math.ceil(pad_h), math.floor(pad_h)
            left, right = math.ceil(pad_w), math.floor(pad_w)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
            return img

        if self.is_training:
            if random.random() < 0.5:
                base, maxh, maxw = self.img_h, self.img_h, self.img_w
                h, w = random.randint(base, maxh), random.randint(base, maxw)
                return _resize_ratio(img, h / w)
            else:
                return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio
        else:
            return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio

    def resize(self, img):
        if self.multiscales:
            return self.resize_multiscales(img, cv2.BORDER_REPLICATE)
        else:
            return cv2.resize(img, (self.img_w, self.img_h))

    def get(self, idx):
        data = json.loads(self.data[idx])
        im = data['filename']
        label = data['text']
        # image = Image.open(os.path.join(self.f, im)).convert(self.convert_mode)
        return os.path.join(self.src, im), label, idx

    def _process_training(self, image):
        if self.data_aug:
            image = self.augment_tfs(image)
        image = self.resize(np.array(image))
        return image

    def _process_test(self, image):
        return self.resize(np.array(image))

    def __getitem__(self, idx):
        image, text, idx_new = self.get(idx)
        return image, text


if __name__ == '__main__':
    src_dir = "/media/palm/Data/ocr/"
    train_jsonl = '/home/palm/PycharmProjects/capocr/data/train.jsonl'
    val_jsonl = '/home/palm/PycharmProjects/capocr/data/val.jsonl'
    train_set = ImageDataset(
        src_dir,
        train_jsonl
    )
    for i in range(len(train_set)):
        train_set[i]
    valid_set = ImageDataset(
        src_dir,
        val_jsonl
    )
    for i in range(len(valid_set)):
        valid_set[i]
