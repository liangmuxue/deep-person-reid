import os
import sys
from PIL import Image
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def default_loader(path):
    return Image.open(path).convert('RGB')


class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        images = []
        labels = open(label).readlines()
        for line in labels:
            items = line.split()
            img_name = items.pop(0)
            if os.path.isfile(os.path.join(root, img_name)):
                cur_label = tuple([int(v) for v in items])
                images.append((img_name, cur_label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')
        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label), img_name

    def __len__(self):
        return len(self.images)


attr_nums = {}
attr_nums['pa100k'] = 26
attr_nums['rap'] = 51
attr_nums['rap2'] = 110
attr_nums['peta'] = 35

description = {}
description['pa100k'] = ['Female',
                         'AgeOver60',
                         'Age18-60',
                         'AgeLess18',
                         'Front',
                         'Side',
                         'Back',
                         'Hat',
                         'Glasses',
                         'HandBag',
                         'ShoulderBag',
                         'Backpack',
                         'HoldObjectsInFront',
                         'ShortSleeve',
                         'LongSleeve',
                         'UpperStride',
                         'UpperLogo',
                         'UpperPlaid',
                         'UpperSplice',
                         'LowerStripe',
                         'LowerPattern',
                         'LongCoat',
                         'Trousers',
                         'Shorts',
                         'Skirt&Dress',
                         'boots']

description['peta'] = ['Age16-30',
                       'Age31-45',
                       'Age46-60',
                       'AgeAbove61',
                       'Backpack',
                       'CarryingOther',
                       'Casual lower',
                       'Casual upper',
                       'Formal lower',
                       'Formal upper',
                       'Hat',
                       'Jacket',
                       'Jeans',
                       'Leather Shoes',
                       'Logo',
                       'Long hair',
                       'Male',
                       'Messenger Bag',
                       'Muffler',
                       'No accessory',
                       'No carrying',
                       'Plaid',
                       'PlasticBags',
                       'Sandals',
                       'Shoes',
                       'Shorts',
                       'Short Sleeve',
                       'Skirt',
                       'Sneaker',
                       'Stripes',
                       'Sunglasses',
                       'Trousers',
                       'Tshirt',
                       'UpperOther',
                       'V-Neck']

description['rap'] = ['Female',
                      'AgeLess16',
                      'Age17-30',
                      'Age31-45',
                      'BodyFat',
                      'BodyNormal',
                      'BodyThin',
                      'Customer',
                      'Clerk',
                      'BaldHead',
                      'LongHair',
                      'BlackHair',
                      'Hat',
                      'Glasses',
                      'Muffler',
                      'Shirt',
                      'Sweater',
                      'Vest',
                      'TShirt',
                      'Cotton',
                      'Jacket',
                      'Suit-Up',
                      'Tight',
                      'ShortSleeve',
                      'LongTrousers',
                      'Skirt',
                      'ShortSkirt',
                      'Dress',
                      'Jeans',
                      'TightTrousers',
                      'LeatherShoes',
                      'SportShoes',
                      'Boots',
                      'ClothShoes',
                      'CasualShoes',
                      'Backpack',
                      'SSBag',
                      'HandBag',
                      'Box',
                      'PlasticBag',
                      'PaperBag',
                      'HandTrunk',
                      'OtherAttchment',
                      'Calling',
                      'Talking',
                      'Gathering',
                      'Holding',
                      'Pusing',
                      'Pulling',
                      'CarryingbyArm',
                      'CarryingbyHand']

description['rap2'] = ['Femal', 'AgeLess16', 'Age17-30', 'Age31-45', 'Age46-60', 'AgeBiger60', 'BodyFatter', 'BodyFat',
                       'BodyNormal', 'BodyThin', 'BodyThiner', 'Customer', 'Employee', 'hs-BaldHead', 'hs-LongHair',
                       'hs-BlackHair', 'hs-Hat', 'hs-Glasses', 'hs-Sunglasses', 'hs-Muffler', 'hs-Mask', 'ub-Shirt',
                       'ub-Sweater', 'ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp', 'ub-Tight',
                       'ub-ShortSleeve', 'ub-Others', 'ub-ColorBlack', 'ub-ColorWhite', 'ub-ColorGray', 'up-ColorRed',
                       'ub-ColorGreen', 'ub-ColorBlue', 'ub-ColorSilver', 'ub-ColorYellow', 'ub-ColorBrown',
                       'ub-ColorPurple', 'ub-ColorPink', 'ub-ColorOrange', 'ub-ColorMixture', 'ub-ColorOther',
                       'lb-LongTrousers', 'lb-Shorts', 'lb-Skirt', 'lb-ShortSkirt', 'lb-LongSkirt', 'lb-Dress',
                       'lb-Jeans', 'lb-TightTrousers', 'lb-ColorBlack', 'lb-ColorWhite', 'lb-ColorGray', 'lb-ColorRed',
                       'lb-ColorGreen', 'lb-ColorBlue', 'lb-ColorSilver', 'lb-ColorYellow', 'lb-ColorBrown',
                       'lb-ColorPurple', 'lb-ColorPink', 'lb-ColorOrange', 'lb-ColorMixture', 'lb-ColorOther',
                       'shoes-Leather', 'shoes-Sports', 'shoes-Boots', 'shoes-Cloth', 'shoes-Sandals', 'shoes-Casual',
                       'shoes-Other', 'shoes-ColorBlack', 'shoes-ColorWhite', 'shoes-ColorGray', 'shoes-ColorRed',
                       'shoes-ColorGreen', 'shoes-ColorBlue', 'shoes-ColorSilver', 'shoes-ColorYellow',
                       'shoes-ColorBrown', 'shoes-ColorPurple', 'shoes-ColorPink', 'shoes-ColorOrange',
                       'shoes-ColorMixture', 'shoes-ColorOther', 'attachment-Backpack', 'attachment-ShoulderBag',
                       'attachment-HandBag', 'attachment-WaistBag', 'attachment-Box', 'attachment-PlasticBag',
                       'attachment-PaperBag', 'attachment-HandTrunk', 'attachment-Baby', 'attachment-Other',
                       'action-Calling', 'action-StrechOutArm', 'action-Talking', 'action-Gathering',
                       'action-LyingCounter', 'action-Squatting', 'action-Running', 'action-Holding', 'action-Pushing',
                       'action-Pulling', 'action-CarryingByArm', 'action-CarryingByHand', 'action-Other']


def Get_Dataset(experiment, approach):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(hue=.05, saturation=.05),
        # transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
    ])

    if experiment == 'pa100k':
        train_dataset = MultiLabelDataset(root='/home/bavon/model/datasets/pa100k/release_data/release_data',
                                          label='data_list/pa-100k/train_val.txt', transform=transform_train)
        val_dataset = MultiLabelDataset(root='/home/bavon/model/datasets/pa100k/release_data/release_data',
                                        label='data_list/pa-100k/test.txt', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['pa100k'], description['pa100k']
    elif experiment == 'rap':
        train_dataset = MultiLabelDataset(root='/home/bavon/model/datasets/rap/RAP_dataset',
                                          label='data_list/rap/train.txt', transform=transform_train)
        val_dataset = MultiLabelDataset(root='/home/bavon/model/datasets/rap/RAP_dataset',
                                        label='data_list/rap/test.txt', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['rap'], description['rap']
    elif experiment == 'rap2':
        train_dataset = MultiLabelDataset(root='/home/bavon/model/datasets/rap2/RAP_dataset',
                                          label='data_list/rap2/train.txt', transform=transform_train)
        val_dataset = MultiLabelDataset(root='/home/bavon/model/datasets/rap2/RAP_dataset',
                                        label='data_list/rap2/test.txt', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['rap2'], description['rap2']
    elif experiment == 'peta':
        train_dataset = MultiLabelDataset(root='data_path',
                                          label='train_list_path', transform=transform_train)
        val_dataset = MultiLabelDataset(root='data_path',
                                        label='val_list_path', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['peta'], description['peta']
