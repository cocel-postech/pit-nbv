# Author : Doyu Lim (2024)

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class nbvLoss(nn.Module):
    def __init__(self):
        super(nbvLoss, self).__init__()
    
    def forward(self, predictedNBV, gtNBV, scoreDif, dist, optDist=1.0, distWeight=0.1, sdifWeight=0.0):
        '''
        Input
            predictedNBV: NBV from network
            gtNBV : ground truth NBV
            scoreDif : coverage ratio difference(%) between next and current cloud
            dist : distance between nbv position and nearest point of accCloud
        Output
            totalLoss : loss value
        '''

        mseLoss = F.mse_loss(predictedNBV, gtNBV)
        optDist_tensor = torch.full_like(dist, optDist)
        #distLoss = F.mse_loss(dist, optDist_tensor)
        distLoss = F.l1_loss(dist, optDist_tensor)
        sDifLoss = (1 - scoreDif/100).mean()

        totalLoss = mseLoss + distWeight * distLoss + sdifWeight * sDifLoss
        return totalLoss