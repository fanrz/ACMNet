import os
import os.path as osp
import cv2
import numpy as np
import random

import torch
from torch.utils import data

class KittiDataset(data.Dataset):
    def __init__(self, root='./datasets/kitti', data_file='train.list', phase='train', joint_transform=None):
        
        self.root = root
        self.data_file = data_file
        self.files = []
        self.joint_transform = joint_transform
        self.phase = phase
        self.no_gt = False
        print('*'*20)
        print('in class of KittiDataset')
        print('root is',root)
        print('data_file is',data_file)
        print('joint_transform is',joint_transform)
        print('phase is',phase)
        # print('no_gt is',no_gt)
        with open(osp.join(self.root, self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                
                data_info = data.split(' ')

                if len(data_info) == 3:
                    self.files.append({
                        "rgb": data_info[0],
                        "sparse": data_info[1],
                        "gt": data_info[2]
                        })
                else:
                    self.files.append({
                        "rgb": data_info[0],
                        "sparse": data_info[1],
                        })
                    self.no_gt = True
        self.nSamples = len(self.files)
        #  you hhave only 98 samples
        print('nSamples is',len(self.files))

        
    def __len__(self):
        return self.nSamples
    
    def read_calib_file(self, path):
        # taken from https://github.com/hunse/kitti
        float_chars = set("0123456789.e+- ")
        data = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(' '))))
                    except ValueError:
                        # casting error: data[key] already eq. value, so pass
                        pass
        print('in the read_calib_file, data is ', data)
        return data

    def read_data(self, index):
        
        sparse = cv2.imread(osp.join(self.root, self.files[index]['sparse']), cv2.IMREAD_UNCHANGED)
        if not self.no_gt:
            gt = cv2.imread(osp.join(self.root, self.files[index]['gt']), cv2.IMREAD_UNCHANGED)
        else:
            gt = sparse
        img = cv2.imread(osp.join(self.root, self.files[index]['rgb']), cv2.IMREAD_COLOR)
        
        h, w = img.shape[0], img.shape[1]

        assert h == gt.shape[0] and w == gt.shape[1]
        assert h == sparse.shape[0] and w == sparse.shape[1]
        # read intrinsics
        if self.phase == 'train':
            calib_dir = self.files[index]['rgb'][0:14]
            print('calib_dir is ', calib_dir)
            cam2cam = self.read_calib_file(osp.join(self.root, calib_dir, 'calib_cam_to_cam.txt'))
            P2_rect = cam2cam['P_rect_02'].reshape(3,4)
            print('here, reshape(3,4) works')
            print('P2_rect is ',P2_rect)
            K = P2_rect[:, :3].astype(np.float32)
            print('in the dataset function, K is ',K)

        elif self.phase in ['val', 'test']:
            calib_name = self.files[index]['sparse'].replace('_velodyne_raw_', '_image_').replace('png', 'txt').replace('velodyne_raw', 'intrinsics')
            with open(osp.join(self.root, calib_name), 'r') as f:
                calib = f.readline()
                calib = calib.splitlines()[0].rstrip().split(' ')
            K = np.zeros((3, 3), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    K[i, j] = float(calib[i*3+j])
        else:
            K = np.zeros((3, 3), dtype=np.float32)

        if  self.no_gt:
            assert w == 1216 and h == 352
        else:
            H = 352 
            s = int(round(w - 1216) / 2)  
            img = img[h-H:, s:s+1216]
            gt = gt[h-H:, s:s+1216]
            sparse = sparse[h-H:, s:s+1216]
            print('H is',H)
            print('s is',s)
            print('img shape is',img.shape)
            print('gt shape is',gt.shape)
            print('sparse shape is',sparse.shape)
            print('K is',K)
            if self.phase == 'train':
                K[0, 2] = K[0, 2] - s
                K[1, 2] = K[1, 2] - (h-H)
                print('after conv, K is',K)
        
        print('img.shape is',img.shape)
        print('sparse.shape is',sparse.shape)
        print('gt.shape is',gt.shape)
        print('K.shape is',K.shape)
        

        return img, gt, sparse, K
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        img, gt, sparse, K = self.read_data(index)

        if self.joint_transform is not None:
            img, gt, sparse  = self.joint_transform((img, gt, sparse, 'kitti'))
        data = {}
        data['img'] = img
        data['gt'] = gt 
        data['sparse'] = sparse
        data['K'] = K

        # print('in the __getitem__ function, the size of data is',data.shape)
        
        return data


def get_dataset(root='./datasets', data_file='train.list', dataset='kitti',
                 phase='train', joint_transform=None):

    return KittiDataset(osp.join(root, dataset), data_file, phase, joint_transform)

