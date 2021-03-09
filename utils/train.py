from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetPlus, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb

wandb.init(name='{}'.format("train_small_data"),
            project="point2control", entity="katefgroup",
           tags=['train_on_small_dataset'],
           job_type='train')


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='ckpts', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', default=True, help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    split='test',
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

P2Control = PointNetPlus(k=2, feature_transform=opt.feature_transform)

if opt.model != '':
    P2Control.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(P2Control.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
P2Control.cuda()

num_batch = len(dataset) // opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        # points should be batch * 3 * 2500
        init_points, final_points, control_type, control_para, init_pose, final_pose = data
        init_points = init_points.transpose(2, 1)
        final_points = final_points.transpose(2, 1)
        init_pose = init_pose[:,-3:]
        final_pose = final_pose[:,-3:]


        init_points, final_points, control_type, control_para, init_pose, final_pose \
        = init_points.type(torch.FloatTensor).cuda(), final_points.type(torch.FloatTensor).cuda(), \
        control_type.type(torch.FloatTensor).cuda(), control_para.type(torch.FloatTensor).cuda(), \
        init_pose.type(torch.FloatTensor).cuda(), final_pose.type(torch.FloatTensor).cuda()

        optimizer.zero_grad()
        P2Control = P2Control.train()

        total_global, control_type_pred, init_pose_pred, final_pose_pred, control_para_pred, trans_feat1, trans_feat2 = P2Control(init_points, final_points)

        vis_img1 = torch.mean(total_global,axis = 0)[:1024].cpu()
        vis_img1 = vis_img1.detach().numpy().reshape(32,32,1)
        v_max, v_min = np.max(vis_img1), np.min(vis_img1)
        vis_img1 =  (255*(vis_img1 - v_min) / (v_max - v_min)).astype(np.uint8)


        vis_img2 = torch.mean(total_global,axis = 0)[1024:].cpu()
        vis_img2 = vis_img2.detach().numpy().reshape(32,32,1)
        v_max, v_min = np.max(vis_img2), np.min(vis_img2)
        vis_img2 =  (255*(vis_img2 - v_min) / (v_max - v_min)).astype(np.uint8)

        #print(pred.size(), target.size())

        total_loss = 0
        # control_type loss
        control_type_loss = F.nll_loss(control_type_pred, torch.squeeze(control_type.type(torch.long)))
        total_loss += control_type_loss
        # transfrom loss
        if opt.feature_transform:
            total_loss += feature_transform_regularizer(trans_feat1.cpu()).cuda() * 0.001
            total_loss += feature_transform_regularizer(trans_feat2.cpu()).cuda() * 0.001

        # mse for init_pose_pred
        init_pose_pred_loss = F.mse_loss(init_pose_pred, init_pose) * 0.1
        total_loss += init_pose_pred_loss
        # mse for final_pose_pred
        final_pose_pred_loss = F.mse_loss(final_pose_pred, final_pose) * 0.1
        total_loss += final_pose_pred_loss
        # mse for control_para_pred
        control_para_pred_loss = F.mse_loss(control_para_pred, control_para) * 5
        total_loss += control_para_pred_loss


        total_loss.backward()
        optimizer.step()
        pred_choice = control_type_pred.data.max(1)[1]
        correct = pred_choice.eq(torch.squeeze(control_type).data).cpu().sum()
        # print('[%d: %d/%d] train loss: %f controller type accuracy: %f' % (epoch, i, num_batch, total_loss.item(), correct.item()/float(opt.batchSize)))

        wandb.log({
            "Train Loss" : total_loss.item(),
            "Train controller type accuracy" :correct.item()/float(opt.batchSize),
            "Train Control_type_loss" : control_type_loss.item(),
            "Train Init_pose_pred_loss" : init_pose_pred_loss.item(),
            "Train Final_pose_pred_loss" : final_pose_pred_loss.item(),
            "Train Control_para_pred_loss" : control_para_pred_loss.item(),
            "Global Feature1": [wandb.Image(vis_img1, caption="Feature")],
            "Global Feature2": [wandb.Image(vis_img2, caption="Feature")],
            })


        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            init_points, final_points, control_type, control_para, init_pose, final_pose = data
            init_points = init_points.transpose(2, 1)
            final_points = final_points.transpose(2, 1)
            init_pose = init_pose[:,-3:]
            final_pose = final_pose[:,-3:]

            init_points, final_points, control_type, control_para, init_pose, final_pose \
            = init_points.type(torch.FloatTensor).cuda(), final_points.type(torch.FloatTensor).cuda(), \
            control_type.type(torch.FloatTensor).cuda(), control_para.type(torch.FloatTensor).cuda(), \
            init_pose.type(torch.FloatTensor).cuda(), final_pose.type(torch.FloatTensor).cuda()

            P2Control = P2Control.eval()
            total_global, control_type_pred, init_pose_pred, final_pose_pred, control_para_pred, trans_feat1, trans_feat2 = P2Control(init_points, final_points)


            test_total_loss = 0
            # control_type loss
            control_type_loss = F.nll_loss(control_type_pred, torch.squeeze(control_type.type(torch.long)))
            test_total_loss += control_type_loss
            # transfrom loss
            if opt.feature_transform:
                test_total_loss += feature_transform_regularizer(trans_feat1.cpu()).cuda() * 0.001
                test_total_loss += feature_transform_regularizer(trans_feat2.cpu()).cuda() * 0.001

            # mse for init_pose_pred
            init_pose_pred_loss = F.mse_loss(init_pose_pred, init_pose) * 0.1
            test_total_loss += init_pose_pred_loss
            # mse for final_pose_pred
            final_pose_pred_loss = F.mse_loss(final_pose_pred, final_pose) * 0.1
            test_total_loss += final_pose_pred_loss
            # mse for control_para_pred
            control_para_pred_loss = F.mse_loss(control_para_pred, control_para)
            test_total_loss += control_para_pred_loss


            pred_choice = control_type_pred.data.max(1)[1]
            correct = pred_choice.eq(torch.squeeze(control_type).data).cpu().sum()
            # print('[%d: %d/%d] %s loss: %f controller type accuracy: %f' % (epoch, i, num_batch, blue('test'), test_total_loss.item(), correct.item()/float(opt.batchSize)))

            wandb.log({
            "Test Loss" : test_total_loss.item(),
            "Test controller type accuracy" :correct.item()/float(opt.batchSize),
            "Test Control_type_loss" : control_type_loss.item(),
            "Test Init_pose_pred_loss" : init_pose_pred_loss.item(),
            "Test Final_pose_pred_loss" : final_pose_pred_loss.item(),
            "Test Control_para_pred_loss" : control_para_pred_loss.item(),
            })
        
    torch.save(P2Control.state_dict(), '%s/P2Control_model_%d.pth' % (opt.outf, epoch))


## test
# shape_ious = []
# for i,data in tqdm(enumerate(testdataloader, 0)):
#     points, target = data
#     points = points.transpose(2, 1)
#     points, target = points.cuda(), target.cuda()
#     P2Control = P2Control.eval()
#     pred, _, _ = P2Control(points)
#     pred_choice = pred.data.max(2)[1]

#     pred_np = pred_choice.cpu().data.numpy()
#     target_np = target.cpu().data.numpy() - 1

#     for shape_idx in range(target_np.shape[0]):
#         parts = range(num_classes)#np.unique(target_np[shape_idx])
#         part_ious = []
#         for part in parts:
#             I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#             U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#             if U == 0:
#                 iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
#             else:
#                 iou = I / float(U)
#             part_ious.append(iou)
#         shape_ious.append(np.mean(part_ious))

# print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))