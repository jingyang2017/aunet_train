import time
import cv2
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from .utils import get_scale_center,get_transform
import socket
host_name = socket.gethostname()

class bp4d_load(data.Dataset):
    def __init__(self, data_name, phase='train', subset=None, transforms=None, scale=260, rect=True,seed=0):
        self.rng = np.random.RandomState(seed=seed)
        self.transform = transforms
        self.phase = phase
        self.rect = rect
        self.scale = scale
        if data_name == 'bp4d':
            if host_name.startswith('learn'):
                predix_path = '/fsx/jinang/Datasets/BP4D/'
            else:
                predix_path = '/media/jd4615/dataB/Datasets/AU/BP4D_jie/'
            self.predix_path = predix_path+'/BP4D-small/'
            data_name = 'BP4D'
            '''
            BP4D_names_label: name of each image path
            BP4D_rect_5pts_label: (rect)+(5 pts): x_lt,y_lt,x_rb,y_rb
            au_labels:array of action units, each line 1 0 1 0 1 1 1 1 0 0 1 1
            '''
            img_list_all = open(predix_path+'/bp4d_small_retina_anno/BP4D_names_label.txt').readlines()
            rect_5pts_list_all = open(predix_path+'/bp4d_small_retina_anno/BP4D_rect_5pts_label.txt').readlines()
            label_list_all = np.loadtxt(predix_path+'/bp4d_small_retina_anno/au_labels.txt')
            if subset==1:
                if self.phase=='train':
                    paths = open('datasets/DAFNet/BP4D_tr1_path.txt').readlines()
                else:
                    paths = open('datasets/DAFNet/BP4D_ts1_path.txt').readlines()
            elif subset==2:
                if self.phase=='train':
                    paths = open('datasets/DAFNet/BP4D_tr2_path.txt').readlines()
                else:
                    paths = open('datasets/DAFNet/BP4D_ts2_path.txt').readlines()
            elif subset==3:
                if self.phase=='train':
                    paths = open('datasets/DAFNet/BP4D_tr3_path.txt').readlines()
                else:
                    paths = open('datasets/DAFNet/BP4D_ts3_path.txt').readlines()
            else:
                raise ValueError()

            self.img_list=[]
            self.rect_5pts_list = []
            self.label_list = []

            start_t = time.time()
            # index in list
            for path_ in paths:
                path_ = path_.replace('_', '/')
                if path_ in img_list_all:
                    index = img_list_all.index(path_)
                    self.img_list.append(img_list_all[index])
                    self.rect_5pts_list.append(rect_5pts_list_all[index])
                    self.label_list.append(label_list_all[index])
            end_t = time.time()
            print('take time',end_t-start_t,'%d/%d'%(len(self.label_list),len(paths)))
            self.label_list = np.asarray(self.label_list)
            assert len(self.img_list)==len(self.rect_5pts_list)==self.label_list.shape[0]

            if self.phase=='train':
                # balance weights
                # this weight is from JAANet. We need to change it for internal data.
                AUoccur_rate = np.zeros((1, self.label_list.shape[1]))
                for i in range(self.label_list.shape[1]):
                    AUoccur_rate[0, i] = sum(self.label_list[:, i] > 0) / float(self.label_list.shape[0])
                AU_weight = 1.0 / AUoccur_rate
                self.AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]
        else:
            raise NotImplementedError()

        self.data_name = data_name

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data ={}
        im_w = 256
        name = self.img_list[index].strip('\n')
        rects, pts5 = self.rect_5pts_list[index].strip('\n').split(';')
        au = self.label_list[index]
        img_name = self.predix_path+name
        img = cv2.imread(img_name)
        if self.rect:
            bb = np.array([float(x) for x in rects.split(',')], dtype=np.float32)
            scale, center = get_scale_center(bb, scale_=self.scale)
            if self.phase == 'train':
                aug_rot = (self.rng.rand() * 2 - 1) * 15  # in deg.
                aug_scale = self.rng.rand() * 0.25 * 2 + (1 - 0.25)  # ex: random_scaling is .25
                scale *= aug_scale
                dx = self.rng.randint(-10 * scale, 10 * scale) / center[0]  # in px
                dy = self.rng.randint(-10 * scale, 10 * scale) / center[1]
            else:
                aug_rot = 0
                dx, dy = 0, 0
            center[0] += dx * center[0]
            center[1] += dy * center[1]
            mat = get_transform(center, scale, (im_w, im_w), aug_rot)[:2]
            img = cv2.warpAffine(img, mat, (im_w, im_w))
        else:
            raise NotImplementedError()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img)
        label = torch.from_numpy(au.astype('float32'))
        data['img'] = img
        data['label'] = label
        return data

