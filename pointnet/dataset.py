from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import open3d as o3d
import pickle


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 split='train',
                 data_augmentation=False):
        self.npoints = npoints
        self.root = root
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.split = split

        self.datapath = os.path.join(self.root, self.split)
        self.folder = os.path.join(self.datapath)

    def __getitem__(self, index):
        #### TODO
        if self.split == "test":
            index += 351
        datapath = os.path.join("/home/sirdome/Documents/HaoZhu/cube_data", self.split)
        folder = os.path.join(datapath, "{:0>5d}".format(index))
        files = sorted(os.listdir(folder))

        f_o = open(os.path.join(folder, files[0]), 'rb')
        dic = pickle.load(f_o)
        pcd_1 = o3d.io.read_point_cloud(os.path.join(folder,files[1]))
        pcd_2 = o3d.io.read_point_cloud(os.path.join(folder,files[2]))

        pcd_1 = np.asarray(pcd_1.points)
        choice = np.random.choice(len(pcd_1), self.npoints, replace=True)
        point_set1 = pcd_1[choice, :]
        point_set1 = point_set1 - np.expand_dims(np.mean(point_set1, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set1 ** 2, axis=1)), 0)
        point_set1 = point_set1 / dist  # scale

        pcd_2 = np.asarray(pcd_2.points)
        choice = np.random.choice(len(pcd_2), self.npoints, replace=True)
        point_set2 = pcd_2[choice, :]

        point_set2 = point_set2 - np.expand_dims(np.mean(point_set2, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set2 ** 2, axis=1)), 0)
        point_set2 = point_set2 / dist  # scale

        point_set_initial = torch.from_numpy(point_set1)
        point_set_final = torch.from_numpy(point_set2)
        if (dic["controller_type"] == "picknplace"):
            controller_type = np.array([0])
        else:
            controller_type = np.array([1])
        controller_type = torch.from_numpy(controller_type)

        controller_para = torch.from_numpy(np.asarray(dic["controller_para"]))
        position_initial = torch.from_numpy(dic["init_pose"])
        position_final = torch.from_numpy(dic["final_pose"])
        return point_set_initial, point_set_final, controller_type, controller_para, position_initial, position_final
    def __len__(self):
        return len(os.listdir(self.folder))
if __name__ == '__main__':
    dataset = "shapenet"
    datapath = "/home/sirdome/Documents/HaoZhu/cube_data"
    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath)
        # print(len(d))
        point_set_initial, point_set_final, controller_type, controller_para, position_initial, position_final = d[0]
        print(point_set_initial, point_set_final, controller_type, controller_para, position_initial, position_final)
        # get_segmentation_classes(datapath)