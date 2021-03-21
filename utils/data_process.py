import os
import sys
import numpy as np
import open3d as o3d
import pickle

class Dataset(object):
    def __init__(self, split, path, start, end):
        self.split = split
        self.All = []
        self.subset = []
        for i in range(start, end+1):
            self.subset.append(os.path.join(path, "data_{}".format(i)))
        # print(self.split, self.subset)
    
    def __len__(self):
        return len(self.All)

    def index(self):
        for i in self.subset:
            pts_list = sorted(os.listdir(os.path.join(i, 'points')), key=lambda x: (int(x[:-4].split('_')[3])))
            pickle_list = sorted(os.listdir(os.path.join(i, 'states')), key=lambda x: (int(x[:-7].split('_')[3])))
            npy_list = sorted(os.listdir(os.path.join(i, 'label')), key=lambda x: (int(x[:-4].split('_')[1])))
            # print(pts_list[0],pts_list[-1])
            # print(pickle_list[0],pickle_list[-1])
            # print(npy_list[0],npy_list[-1])
            
            for j in range( len(pickle_list) ):
                item = {}
                item['pts'] = [ pts_list[j*2], pts_list[j*2+1] ]
                item['pickle'] = pickle_list[j]
                item['npy'] = npy_list[ int(pickle_list[j].split('_')[1]) ]
                succ = np.load(os.path.join(i, 'label', item['npy']))[0]
                if (succ == 0):
                    continue
                else:
                    abs_item={}
                    abs_item['pts'] = [ os.path.join(i, 'points', item['pts'][0]), os.path.join(i, 'points', item['pts'][1]) ]
                    abs_item['pickle'] = os.path.join(i, 'states', item['pickle'])
                    abs_item['npy'] = os.path.join(i, 'label', item['npy'])
                    self.All.append(abs_item)

    def save(self):
        for i in range(len(self)):
            folder = os.path.join(path, 'data_Mar13', self.split, str(i))
            os.system( 'mkdir -vp {}'.format(folder))
            os.system( 'cp {} {} {}'.format(self.All[i]['pts'][0], self.All[i]['pts'][1], folder) )
            os.system( 'cp {} {}'.format(self.All[i]['pickle'], folder) )
            os.system( 'cp {} {}'.format(self.All[i]['npy'], folder) )

if __name__ == "__main__":
    path = '/home/sirdome/Documents/HaoZhu'
    Train = Dataset('train', path, 0, 8)
    Train.index()
    Train.save()

    Test = Dataset('test', path, 9, 10)
    Test.index()
    Test.save()
