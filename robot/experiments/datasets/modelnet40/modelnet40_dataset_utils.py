#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate  %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if isinstance(sample, dict):
            return {item: ToTensor()(sample[item]) for item in sample}
        else:
            n_tensor = torch.tensor(sample)
            return n_tensor



def transfer_into_support_format(item_id, pointcloud1, pointcloud2, R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba):
    pointcloud1, pointcloud2, R_ab, \
    translation_ab, R_ba, translation_ba, \
    euler_ab, euler_ba = pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
                         translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
                         euler_ab.astype('float32'), euler_ba.astype('float32')

    extra_info=  {"rotation_ab": R_ab, "translation_ab": translation_ab, "rotation_ba": R_ba,
                  "translation_ba": translation_ba, "euler_ab":euler_ab, "euler_ba":euler_ba,
                  "gt_flow":pointcloud2.T - pointcloud1.T, "has_gt":True,"corr_source_target":True}
    source_dict = {"points":pointcloud1.T, "extra_info":extra_info}
    target_dict = {"points": pointcloud2.T, "extra_info":extra_info}
    transform = ToTensor()
    source_dict = {key: transform(fea) for key, fea in source_dict.items()}
    target_dict = {key: transform(fea) for key, fea in target_dict.items()}
    pair_name = "{:07d}".format(item_id +1)
    return {
        "source":source_dict,
        "target":target_dict,
        "pair_name": pair_name,
        "source_info":{},
        "target_info":{}
    }


class ModelNet40(Dataset):
    def __init__(self, data_path, dataset_opt,  phase='train'):
        super(ModelNet40, self).__init__()
        self.phase = phase
        partition = phase
        category = None
        num_points  = dataset_opt[("num_points", 1024, "num of points")]
        num_subsampled_points  = dataset_opt[("num_subsampled_points", 768, "num of subsampled points")]
        gaussian_noise  = dataset_opt[("gaussian_noise", False, "'Wheter to add gaussian noise")]
        unseen = dataset_opt[("unseen", False, "Whether to test on unseen category")]
        rot_factor = dataset_opt[("rot_factor", 4, "Divided factor of rotation")]
        partial_only_during_test = dataset_opt[("partial_only_during_test", False, "run paritial registration during test")]
        # to be compatible to robot interface
        if partition=="val":
            partition = "test"
        if partition=="debug":
            partition = "train"
        self.data, self.label = load_data(partition)
        if category is not None:
            self.data = self.data[self.label==category]
            self.label = self.label[self.label==category]
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.rot_factor = rot_factor
        self.subsampled = False
        if partition=="train":
            self.subsampled=False
        else:
            if partial_only_during_test:
                if partition=="test":
                    self.subsampled = True
            elif num_points != num_subsampled_points:
                    self.subsampled = True


        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        if self.phase =="test":
            pointcloud1 = np.random.permutation(pointcloud1.T).T
            pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)
        return transfer_into_support_format(item, pointcloud1,
                                             pointcloud2, R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba)

    def __len__(self):
        return self.data.shape[0]







if __name__ == '__main__':
    print('hello world')
