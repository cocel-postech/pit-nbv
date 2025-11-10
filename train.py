# Author : Doyu Lim (2024)
# Train PCT-V

import argparse
import os
import csv
import time
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import profiler
from torch.optim.lr_scheduler import StepLR

from dataloader_poisson import Poisson_dataset
from util.loss import nbvLoss
from network.pct_v import PointTransformer_View


def print_data(i_batch, model_id, gt_pcd, partial_pcd, nbv, score, iter, poseList, scoreList, nbvC, sDif, dist):
    print('\n--------------------------------- data view ---------------------------------\n')
    print('batch idx \t', i_batch)
    print('model_id \t', model_id.shape, model_id)      # Batch X 1
    print('gt_pcd \t', gt_pcd.shape, gt_pcd[0][0])      # Batch X 5000 X 3
    print('partial_pcd    \t', partial_pcd.shape)       # Batch X (# of gt points) X 3/6
    print('nbv_pose \t', nbv.shape)                     # Batch X 6
    print('nbv scroe \t', score.shape)                  # Batch X 1
    print('iteration \t', iter.shape)                   # Batch X 10
    print('all pose\t', poseList.shape)                 # Batch X 11 X 6
    print('all score\t', scoreList.shape)               # Batch X 11 X 1
    print('nbv candidate\t', nbvC.shape)                # Batch X 10 X 6
    print('score Diff\t', sDif.shape)                   # Batch X 1
    print('near distance\t', dist.shape)                # Batch X 1
    print('\n-----------------------------------------------------------------------------\n\n')

def train(args, model, device, train_dataloader, optimizer, criterion, epoch):
    model.train()

    steps = 0

    csv_file = open(os.path.join(args.log_dir, 'train_loss.csv'), 'a+')
    csv_writer = csv.writer(csv_file)

    for i_batch, batch_data in enumerate(train_dataloader):

        start_time = time.time()

        # model, gtPoints, partialPoints, curnbv, curScore, oneHotIter, pose, score, nbvCandidate, scoreDif
        
        i_batch = int(i_batch)+1
        batch_model_id = np.array(batch_data[0])
        batch_gt_pcd = np.array(batch_data[1])
        batch_partial_pcd = np.array(batch_data[2])
        batch_nbv = np.array(batch_data[3])
        batch_score = np.array(batch_data[4])
        batch_iter = np.array(batch_data[5])
        batch_poseList = np.array(batch_data[6])
        batch_scoreList = np.array(batch_data[7])
        batch_nbvC = np.array(batch_data[8])
        batch_sDif = np.array(batch_data[9])
        batch_dist = np.array(batch_data[10])
        
        # print_data(i_batch, batch_model_id, batch_gt_pcd, batch_partial_pcd, batch_nbv, batch_score, batch_iter, \
        #           batch_poseList, batch_scoreList, batch_nbvC, batch_sDif, batch_dist)

        batch_partial_pcd = torch.from_numpy(batch_partial_pcd).to(device).float()
        batch_nbv = torch.from_numpy(batch_nbv).to(device).float()
        batch_iter = torch.from_numpy(batch_iter).to(device)
        batch_score = torch.from_numpy(batch_score).to(device).float()
        batch_sDif = torch.from_numpy(batch_sDif).to(device).float()
        batch_dist = torch.from_numpy(batch_dist).to(device).float()

        optimizer.zero_grad()
        Snet = model(batch_partial_pcd, device=device) # network
        
        loss = criterion(Snet, batch_nbv, batch_sDif, batch_dist, optDist=1.0, distWeight=args.distWeight, sdifWeight=args.sdifWeight)
        loss.backward()
        optimizer.step()

        end_time = time.time()
        res_time = end_time - start_time

        print('Train Epoch: {} [ Batch {}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.2f}'.format(
            epoch, i_batch, len(train_dataloader), 100. * i_batch / len(train_dataloader), loss.item(), res_time))
        
        if i_batch == 1:
            csv_writer.writerow(['epoch', 'batch index', 'train loss'])
            losses = []
        csv_writer.writerow([epoch, i_batch, loss.item()])
        losses.append(loss.item())
        
        if args.viz:
            plt.clf()
            plt.plot(losses, label='Training Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.legend()
            plt.pause(0.05)

def valid(args, model, device, valid_dataloader, criterion, epoch):

    model.eval()
    valid_loss = 0

    csv_file = open(os.path.join(args.log_dir, 'valid_loss.csv'), 'a+')
    csv_writer = csv.writer(csv_file)

    with torch.no_grad():
        for batch_data in valid_dataloader:
            
            position = []
            batch_model_id = np.array(batch_data[0])
            batch_gt_pcd = np.array(batch_data[1])
            batch_partial_pcd = np.array(batch_data[2])
            batch_nbv = np.array(batch_data[3])
            batch_score = np.array(batch_data[4])
            batch_iter = np.array(batch_data[5])
            batch_poseList = np.array(batch_data[6])
            batch_scoreList = np.array(batch_data[7])
            batch_nbvC = np.array(batch_data[8])
            batch_sDif = np.array(batch_data[9])
            batch_dist = np.array(batch_data[10])

            batch_partial_pcd = torch.from_numpy(batch_partial_pcd).to(device).float()
            batch_nbv = torch.from_numpy(batch_nbv).to(device).float()
            batch_score = torch.from_numpy(batch_score).to(device).float()
            batch_sDif = torch.from_numpy(batch_sDif).to(device).float()
            batch_dist = torch.from_numpy(batch_dist).to(device).float()

            Snet = model(batch_partial_pcd, device=device) # network
            valid_loss += criterion(Snet, batch_nbv, batch_sDif, batch_dist, optDist=1.0, distWeight=args.distWeight, sdifWeight=args.sdifWeight).item()

    valid_loss /= len(valid_dataloader)
    
    print('\nValid set: Average loss: {:.4f}'.format(valid_loss))
    if epoch == 1:
        csv_writer.writerow(['epoch', 'valid loss'])
    csv_writer.writerow([epoch, valid_loss])

def main(args):

    # Dataset
    shapenet_dataset_training = Poisson_dataset(dataset_path=args.data_path, mode='training', \
                                inputSample=args.sample_input, gtSample=args.sample_gt)
    shapenet_dataset_validation = Poisson_dataset(dataset_path=args.data_path, mode='validation', \
                                inputSample=args.sample_input, gtSample=args.sample_gt)
    # shapenet_dataset_test = Poisson_dataset(dataset_path=args.data_path, mode='test', \
    #                           inputSample=args.sample_input, gtSample=args.sample_gt)

    # Dataloader
    train_dataloader = DataLoader(dataset=shapenet_dataset_training, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(dataset=shapenet_dataset_validation, batch_size=args.batch_size, shuffle=False,num_workers=0)
    # test_dataloader = DataLoader(dataset=shapenet_dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Set GPU
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')   #cuda machine 환경이면 "cuda"
    torch.cuda.set_device(device)
    print('[device]', device)

    # Model
    model = PointTransformer_View(in_dim=3, out=args.output).to(device)
    model_file = 'pct_v'

    if not args.load_model:
        if os.path.exists(args.log_dir):
            delete_key = input('%s directory already exists. Delete? [y (or enter)/N]' % args.log_dir)
            if delete_key == 'y' or delete_key == "":
                os.system('rm -rf %s/*' % args.log_dir)
            if delete_key == 'N':
                return
        os.makedirs(os.path.join(args.log_dir, 'model'))
        with open(os.path.join(args.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(args)):
                log.write(arg + ': ' + str(getattr(args, arg)) + '\n') # log of arguments
        os.system('cp network/%s.py %s' % (model_file, args.log_dir))  # backup of model definition
    else:
        print('Load trained model', args.load_model)
        model.load_state_dict(torch.load(os.path.join(args.log_dir,args.load_model)))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.decay_steps, gamma=args.decay_rate)
    criterion = nbvLoss()

    for epoch in range(args.start_epoch, args.epochs + 1):
        start_time = time.time()
        train(args, model, device, train_dataloader, optimizer, criterion, epoch)
        train_time = time.time()
        print(f'[Epoch {epoch} Training Time] {train_time-start_time} sec')
        valid(args, model, device, valid_dataloader, criterion, epoch)
        print(f'[Epoch {epoch} Validation Time] {time.time()-train_time}\n')
        print(f'[Epoch {epoch} Total Time] {time.time()-start_time} sec\n')

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'[lr] {current_lr}')
        
        if epoch % args.save_term == 0:
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'model', 'epoch_'+str(epoch)+'.pt'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/media/owner/CoCEL/nbv_dataset/owner_poisson_overlap_ShapeNet_nbv_dataset_scale_10_sDif_1.hdf5')
    parser.add_argument('--log_dir', default='log/240902')
    parser.add_argument('--save_term', type=int, default=100)
    parser.add_argument('--viz', type=bool, default=False)

    # training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', type=int, default=6)
    parser.add_argument('--sample_input', type=int, default=1024)
    parser.add_argument('--sample_gt', type=int, default=5000)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--decay_rate', type=float, default=1)
    parser.add_argument('--decay_steps', type=float, default=50000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load_model', default=False) # default=False, 'model/epoch_5.pt'
    parser.add_argument('--start_epoch', default=1)    # default=1      (loaded model's epoch + 1)

    # loss function weight
    parser.add_argument('--distWeight', type=float, default=0.1)
    parser.add_argument('--sdifWeight', type=float, default=0.0)

    args = parser.parse_args()

    main(args)