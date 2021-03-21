import numpy as np
import os
import sys
import open3d as o3d
import torch

def readnp(path):
    succ = 0
    fail = 0
    files = sorted(os.listdir(path))
    for i in files:
        if i[-3:] != "npy":
            continue
        f = os.path.join(path, i)
        n = np.load(f)
        # print(f, n)
        if (n[0] == 1):
            succ += 1
        else:
            fail += 1
    print('succ=', succ, 'fail=', fail, succ/(succ+fail))
    return  succ/(succ+fail)

# def readpoints():
#     path = "/Users/zhu/Desktop/cube_data/rollout_0_picknplace_1.ply"
#     # pcd = o3d.geometry.PointCloud()
#     pcd = o3d.io.read_point_cloud(path)
#     print(pcd)
#     print(np.asarray(pcd.colors)[0])
#     o3d.io.write_point_cloud("/Users/zhu/Desktop/test.pts", pcd)

if __name__ == "__main__":
    path ="/home/sirdome/Documents/HaoZhu/data_0/label"
    readnp(path)