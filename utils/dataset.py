from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import os


class VOCDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        self._len = len(self.paths)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        path, mask_path = self.paths[index]
        image = np.array(Image.open(path))

        # mask_path = os.path.join(VOC_MASK_PATH, path.split('/')[-1].split('.')[0] + '.png')
        if not os.path.exists(mask_path):
            return 'mask not found'
        mask = np.array(Image.open(mask_path))
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        image = torch.from_numpy(np.array(image, dtype=np.float)).permute(2, 0, 1)
        image = image.type(torch.FloatTensor)
        mask = torch.from_numpy(np.array([mask], dtype=np.uint8))
        return image / 255, mask


class COCODataset(Dataset):
    def __init__(self, coco_ids, anno, transform=None):
        self.coco_ids = coco_ids
        self.coco = anno
        self.transform = transform
        self._len = len(self.coco_ids)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        idx = self.coco_ids[index]
        img = self.coco.imgs[idx]
        image = np.array(Image.open(os.path.join('/content/datasets/coco/train2017', img['file_name'])))

        cat_ids = self.coco.getCatIds()
        # if 1 in cat_ids:
        #     cat_ids = [1]
        anns_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)
        if len(anns) == 0:
            mask = np.zeros(image.shape[:2])
        else:
            mask = self.coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += self.coco.annToMask(anns[i])
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        if len(torch.from_numpy(np.array(image, dtype=np.float)).shape) == 2:
            img = torch.unsqueeze(torch.from_numpy(np.array(image, dtype=np.float)), dim=0)
            image = torch.cat([img, img, img])
        else:
            image = torch.from_numpy(np.array(image, dtype=np.float)).permute(2, 0, 1)
        image = image.type(torch.FloatTensor)
        mask = torch.from_numpy(np.array([mask], dtype=np.uint8))
        return image / 255, mask
