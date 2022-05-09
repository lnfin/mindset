import os
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import albumentations as A
from pycocotools.coco import COCO
from PIL import Image
import xml.etree.ElementTree as ET
from .dataset import VOCDataset, COCODataset


def get_paths(data_folder):
    """
    :param dirs: list of directories or one directory, where data is
    :return: [[image_path, mask_path], ...]
    """

    voc_image_path = os.path.join(data_folder, 'JPEGImages')
    voc_mask_path = os.path.join(data_folder, 'SegmentationObject')
    voc_anno_path = os.path.join(data_folder, 'Annotations')

    paths = [os.path.join(voc_mask_path, x) for x in os.listdir(voc_mask_path)]
    labels = set()
    cor_mask_paths = []
    for path in paths:
        root = ET.parse(os.path.join(voc_anno_path, path.split('/')[-1].split('.')[0] + '.xml')).getroot()
        # if root[-1][0].text != 'person':
        #     continue
        if 'VOC' in root[-1][0].text:
            continue
        im = np.array(Image.open(path))
        im = np.where(im > 0, 1, 0)
        if (np.sum(im) / np.sum(np.ones(im.shape))) > 0.3:
            cor_mask_paths.append(path)

    image_with_mask = []
    for mask_path in cor_mask_paths:
        image_path = os.path.join(voc_image_path, mask_path.split('/')[-1].split('.')[0] + '.jpg')
        image_with_mask.append([image_path, mask_path])
    # # assert 1==2, voc_image_path
    # voc_images = [os.path.join(voc_image_path, x) for x in os.listdir(voc_image_path)]
    # voc_images_with_masks = []
    # for path in voc_images:
    #     mask_path = os.path.join(voc_mask_path, path.split('/')[-1].split('.')[0] + '.png')
    #     if os.path.exists(mask_path):
    #         voc_images_with_masks.append([path, mask_path])
    return image_with_mask


def filter_coco_ids(coco, coco_ids):
    cor_ids = []
    for coco_id in coco_ids:
        img = coco.imgs[coco_id]
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=coco.getCatIds(), iscrowd=None)
        if len(anns_ids) != 1:
            continue
        anns = coco.loadAnns(anns_ids)
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)
        if not 0.3 < np.sum(mask) / np.sum(np.ones(mask.shape)) < 0.9:
            continue
        cor_ids.append(coco_id)
    return cor_ids


def coco_data_generator(cfg):
    if cfg.coco_annotations:
        coco = COCO(cfg.coco_annotations)
        coco_ids = list(coco.imgs.keys())
        coco_ids = filter_coco_ids(coco, coco_ids)
        image_paths = np.asarray(coco_ids)

    train_paths, val_paths = [], []
    if not cfg.kfold:
        _train_paths, _val_paths = train_test_split(image_paths, test_size=cfg.val_size, random_state=cfg.seed)
    else:
        kf = KFold(n_splits=cfg.n_splits)
        for i, (train_index, val_index) in enumerate(kf.split(image_paths)):
            if i + 1 == cfg.fold_number:
                _train_paths = image_paths[train_index]
                _val_paths = image_paths[val_index]


    for paths in _train_paths:
        train_paths.append(paths.tolist())
    for paths in _val_paths:
        val_paths.append(paths.tolist())
    random.shuffle(train_paths)
    random.shuffle(val_paths)
    return train_paths, val_paths


def voc_data_generator(cfg):
    image_paths = get_paths(cfg.voc_data_folder)
    image_paths = np.asarray(image_paths)
    train_paths, val_paths = [], []

    if not cfg.kfold:
        _train_paths, _val_paths = train_test_split(image_paths, test_size=cfg.val_size, random_state=cfg.seed)
    else:
        kf = KFold(n_splits=cfg.n_splits)
        for i, (train_index, val_index) in enumerate(kf.split(image_paths)):
            if i + 1 == cfg.fold_number:
                _train_paths = image_paths[train_index]
                _val_paths = image_paths[val_index]

    for paths in _train_paths:
        train_paths.append(paths.tolist())
    for paths in _val_paths:
        val_paths.append(paths.tolist())
    random.shuffle(train_paths)
    random.shuffle(val_paths)
    return train_paths, val_paths


def get_transforms(cfg):
    # getting transforms from albumentations
    pre_transforms = [getattr(A, item["name"])(**item["params"]) for item in cfg.pre_transforms]
    augmentations = [getattr(A, item["name"])(**item["params"]) for item in cfg.augmentations]
    post_transforms = [getattr(A, item["name"])(**item["params"]) for item in cfg.post_transforms]

    # concatenate transforms
    train = A.Compose(pre_transforms + augmentations + post_transforms)
    test = A.Compose(pre_transforms + post_transforms)
    return train, test


def get_loaders(cfg):
    # getting transforms
    train_transforms, test_transforms = get_transforms(cfg)

    # getting train and val paths
    coco_train, coco_val = coco_data_generator(cfg)
    voc_train, voc_val = voc_data_generator(cfg)

    if cfg.coco_annotations:
        coco = COCO(cfg.coco_annotations)
    # creating datasets
    coco_train_ds = COCODataset(coco_ids=coco_train, anno=coco, transform=train_transforms)
    coco_val_ds = COCODataset(coco_ids=coco_val, anno=coco, transform=train_transforms)
    voc_train_ds = VOCDataset(voc_train, transform=train_transforms)
    voc_val_ds = VOCDataset(voc_val, transform=train_transforms)
    coco_voc_train_ds = torch.utils.data.ConcatDataset([coco_train_ds, voc_train_ds])
    coco_voc_val_ds = torch.utils.data.ConcatDataset([coco_val_ds, voc_val_ds])

    # creating data loaders
    train_dl = DataLoader(coco_voc_train_ds, shuffle=True, batch_size=cfg.batch_size, drop_last=True)
    val_dl = DataLoader(coco_voc_val_ds, shuffle=True, batch_size=cfg.batch_size, drop_last=True)
    return train_dl, val_dl
