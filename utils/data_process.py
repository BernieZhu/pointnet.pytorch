import numpy as np
import os
import sys
import open3d as o3d
import pickle

def renamePickle(pickle_list):
    for i in (pickle_list):
        old_path = os.path.join(pickle_dir, i)
        segment = i[:-7].split("_", 3)
        new_name = "{:0>5d}_{}_{:0>5d}.pickle".format(int(segment[1]), segment[2], int(segment[3]))
        new_path = os.path.join(pickle_dir, new_name)
        os.system( "mv {} {}".format(old_path, new_path))

def renamePts(pts_list):
    for i in (pts_list):
        old_path = os.path.join(pts_dir, i)
        segment = i[:-4].split("_", 3)
        new_name = "{:0>5d}_{}_{:0>5d}.pts".format(int(segment[1]), segment[2], int(segment[3]))
        new_path = os.path.join(pts_dir, new_name)
        os.system( "mv {} {}".format(old_path, new_path))

def movePickle(path):
    i = 0
    pts1 = 0
    pts2 = 0
    while (pts1 <= 778): 
        n = "{:0>5d}".format(i)
        new_folder = os.path.join(path, "train", n)
        os.system("mkdir {}".format(new_folder))
        i += 1
        pts1 = pts2 + 1
        pts2 = pts1 + 1
        # os.system( "ls *{:0>5d}.pts *{:0>5d}.pts".format(pts1, pts2))
        os.system( "mv *{:0>5d}.pts *{:0>5d}.pts {}".format(pts1, pts2, new_folder))

def movePts(path):
    i = 1
    while (i <= 390): 
        n = "{:0>5d}".format(i)
        f = "{:0>5d}".format(i-1)
        new_folder = os.path.join(path, "train", f)
        os.system( "mv *{}.pickle {}".format(n, new_folder))
        i += 1

if __name__ == '__main__':
    path = "/home/sirdome/Documents/HaoZhu/data"
    pts_dir = os.path.join(path, "points")
    pickle_dir = os.path.join(path, "states")
    pts_list = os.listdir(pts_dir)
    pickle_list = os.listdir(pickle_dir)
    
    renamePickle(pickle_list)
    renamePts(pts_list)
    movePickle(path)
    movePts(path)