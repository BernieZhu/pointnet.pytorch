from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetPlusfeat(nn.Module):
    def __init__(self, feature_transform = False):
        super(PointNetPlusfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        # self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x, pointfeat, trans_feat
        # if self.global_feat:
        #     return x, trans, trans_feat
        # else:
        #     x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        #     return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

class PointNetPlus(nn.Module):
    def __init__(self, k=2, feature_transform=True):
        super(PointNetPlus, self).__init__()
        self.feature_transform = feature_transform

        self.feat1 = PointNetPlusfeat(feature_transform=feature_transform)
        self.feat2 = PointNetPlusfeat(feature_transform=feature_transform)

        self.fc1_1 = nn.Linear(2048, 1024)
        self.fc1_2 = nn.Linear(1024, 512)
        self.fc1_3 = nn.Linear(512, 256)
        self.fc1_4 = nn.Linear(256, 128)
        self.fc1_5 = nn.Linear(128, 64)
        self.fc1_6 = nn.Linear(64, k)

        self.bn1_1 = nn.BatchNorm1d(1024)
        self.bn1_2 = nn.BatchNorm1d(512)
        self.bn1_3 = nn.BatchNorm1d(256)
        self.bn1_4 = nn.BatchNorm1d(128)
        self.bn1_5 = nn.BatchNorm1d(64)

        self.fc2_1 = nn.Linear(2048, 1024)
        self.fc2_2 = nn.Linear(1024, 512)
        self.fc2_3 = nn.Linear(512, 256)
        self.fc2_4 = nn.Linear(256, 128)
        self.fc2_5 = nn.Linear(128, 64)
        self.fc2_6 = nn.Linear(64, k)

        self.bn2_1 = nn.BatchNorm1d(1024)
        self.bn2_2 = nn.BatchNorm1d(512)
        self.bn2_3 = nn.BatchNorm1d(256)
        self.bn2_4 = nn.BatchNorm1d(128)
        self.bn2_5 = nn.BatchNorm1d(64)

        self.conv1_1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, 128, 1)
        self.conv1_4 = torch.nn.Conv1d(128, 3, 1)
        self.bnc1_1 = nn.BatchNorm1d(512)
        self.bnc1_2 = nn.BatchNorm1d(256)
        self.bnc1_3 = nn.BatchNorm1d(128)

        self.conv2_1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv2_3 = torch.nn.Conv1d(256, 128, 1)
        self.conv2_4 = torch.nn.Conv1d(128, 3, 1)
        self.bnc2_1 = nn.BatchNorm1d(512)
        self.bnc2_2 = nn.BatchNorm1d(256)
        self.bnc2_3 = nn.BatchNorm1d(128)

        self.conv3_1 = torch.nn.Conv1d(2176, 1024, 1)
        self.conv3_2 = torch.nn.Conv1d(1024, 512, 1)
        self.conv3_3 = torch.nn.Conv1d(512, 256, 1)
        self.conv3_4 = torch.nn.Conv1d(256, 128, 1)
        self.conv3_5 = torch.nn.Conv1d(128, 3, 1)
        self.bnc3_1 = nn.BatchNorm1d(1024)
        self.bnc3_2 = nn.BatchNorm1d(512)
        self.bnc3_3 = nn.BatchNorm1d(256)
        self.bnc3_4 = nn.BatchNorm1d(128)


        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        n1_pts = x1.size()[2]
        n2_pts = x2.size()[2]
        # get two global feature and two n * 64pointfeat
        x1, pointfeat1, trans_feat1 = self.feat1(x1)
        x2, pointfeat2, trans_feat2 = self.feat2(x2)

        # concat two global feature
        total_global = torch.cat([x1, x2], 1)

        # Pass total global to FC to predict type 2048 to k

        y_type = F.relu(self.bn1_1(self.fc1_1(total_global)))
        y_type = F.relu(self.bn1_2(self.fc1_2(y_type)))
        y_type = F.relu(self.bn1_3(self.fc1_3(y_type)))
        y_type = F.relu(self.bn1_4(self.fc1_4(y_type)))
        y_type = F.relu(self.bn1_5(self.dropout(self.fc1_5(y_type))))
        y_type = self.fc1_6(y_type)
        y_type = F.log_softmax(y_type, dim=1)

        # Densefeature to 3 * 1 object pose
        x1 = x1.view(-1, 1024, 1).repeat(1, 1, n1_pts)
        part_feat1 = torch.cat([x1, pointfeat1], 1)
        batchsize1 = part_feat1.size()[0]

        # Densefeature to 3 * 1 object pose
        x2 = x2.view(-1, 1024, 1).repeat(1, 1, n2_pts)
        part_feat2 = torch.cat([x2, pointfeat2], 1)
        batchsize2 = part_feat2.size()[0]

        # predict init_pose
        init_pose = F.relu(self.bnc1_1(self.conv1_1(part_feat1)))
        init_pose = F.relu(self.bnc1_2(self.conv1_2(init_pose)))
        init_pose = F.relu(self.bnc1_3(self.conv1_3(init_pose)))
        init_pose = self.conv1_4(init_pose)
        init_pose = torch.mean(init_pose, axis = 2)

        # predict final_pose
        final_pose = F.relu(self.bnc2_1(self.conv2_1(part_feat2)))
        final_pose = F.relu(self.bnc2_2(self.conv2_2(final_pose)))
        final_pose = F.relu(self.bnc2_3(self.conv2_3(final_pose)))
        final_pose = self.conv2_4(final_pose)
        final_pose = torch.mean(final_pose, axis = 2)

        # predict control para from 2 Densefeature
        total_dense = torch.cat([part_feat1, part_feat2], 1)

        control_para = F.relu(self.bnc3_1(self.conv3_1(total_dense)))
        control_para = F.relu(self.bnc3_2(self.conv3_2(control_para)))
        control_para = F.relu(self.bnc3_3(self.conv3_3(control_para)))
        control_para = F.relu(self.bnc3_4(self.conv3_4(control_para)))
        control_para = self.conv3_5(control_para)

        control_para = torch.mean(control_para, axis = 2)

        return total_global, y_type, init_pose, final_pose, control_para, trans_feat1, trans_feat2

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    # pointfeat = PointNetfeat(global_feat=True)
    # out, _, _ = pointfeat(sim_data)
    # print('global feat', out.size())

    # pointfeat = PointNetfeat(global_feat=False)
    # out, _, _ = pointfeat(sim_data)
    # print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
    # seg = PointNetPlus(k = 2)
    # total_global, y_type, init_pose, final_pose, control_para, trans_feat1, trans_feat2 = seg(sim_data, sim_data)
    # print(total_global.shape, y_type.shape, init_pose.shape, final_pose.shape, control_para.shape)



    
