import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn import init, Parameter
import torch.nn.functional as F
from torch.nn import DataParallel
from point_utils import knn_operation, gather_operation, grouping_operation
import numpy as np

def test():
    knn=[1,1,1]
    nsamples=[10, 5, 2]
    K = torch.tensor([[[721.5377,   0.0000, 596.5593],
            [  0.0000, 721.5377, 149.8540],
            [  0.0000,   0.0000,   1.0000]]], device='cuda:0')

    depth = torch.rand(1,1,8,10).cuda()
    # [0.7277, 0.3465, 0.7522, 0.7081, 0.5236, 0.1151, 0.0334, 0.7528, 0.4630, 0.7499],
    # [0.6335, 0.3929, 0.9721, 0.9103, 0.2424, 0.8838, 0.2689, 0.3432, 0.0343, 0.2192],
    # [0.6933, 0.7725, 0.8671, 0.7401, 0.2477, 0.5539, 0.3930, 0.0293, 0.3636, 0.3679],
    # [0.5770, 0.3282, 0.9464, 0.7382, 0.5294, 0.4362, 0.8141, 0.9612, 0.9687, 0.1492],
    # [0.7374, 0.9402, 0.9162, 0.3112, 0.6881, 0.0594, 0.8559, 0.0475, 0.9376, 0.2167],
    # [0.1164, 0.3681, 0.5035, 0.7327, 0.7007, 0.2706, 0.9434, 0.2397, 0.0017, 0.1712],
    # [0.4584, 0.7841, 0.4194, 0.7910, 0.4587, 0.9378, 0.8015, 0.9899, 0.3114, 0.6071],
    # [0.7666, 0.8204, 0.2047, 0.3488, 0.7672, 0.1541, 0.2341, 0.1244, 0.4511, 0.9264]

    print('depth is',depth)
    n, c, h, w = depth.shape
    # n, c, h, w is 1,1,8,10
    # notice (xx,yy) is the coordinate
    xx = torch.arange(0, w).view(1, -1).repeat(h, 1).float().cuda().view(1, 1, h, w).repeat(n, 1, 1, 1)
    # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
    # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
    # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
    # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
    # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
    # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
    # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
    # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    yy = torch.arange(0, h).view(-1, 1).repeat(1, w).float().cuda().view(1, 1, h, w).repeat(n, 1, 1, 1)
    # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    # [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    # [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
    # [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
    # [4., 4., 4., 4., 4., 4., 4., 4., 4., 4.],
    # [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
    # [6., 6., 6., 6., 6., 6., 6., 6., 6., 6.],
    # [7., 7., 7., 7., 7., 7., 7., 7., 7., 7.]
    print('xx is',xx)
    print('yy is',yy)

    spoints = []
    sidxs = []
    nnidxs = []
    masks = []
    print('******** beginning ***********')
    for i in range(1, 4):
        print('i is',i)
        depth, max_ind = F.max_pool2d(depth, kernel_size=2, stride=2, return_indices=True)
        print('depth is',depth)
        # in loop 1,
        # [0.7277, 0.9721, 0.8838, 0.7528, 0.7499],
        # [0.7725, 0.9464, 0.5539, 0.9612, 0.9687],
        # [0.9402, 0.9162, 0.7007, 0.9434, 0.9376],
        # [0.8204, 0.7910, 0.9378, 0.9899, 0.9264]
        print('max_ind is',max_ind)
        # in loop 1,
        # [ 0, 12, 15,  7,  9],
        # [21, 32, 25, 37, 38],
        # [41, 42, 54, 56, 48],
        # [71, 63, 65, 67, 79]
        xy = torch.cat([xx, yy], 1).view(n, 2, -1)
        # in loop 1, torch.Size([1, 2, 80])
        # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5., 6.,
        # 7., 8., 9., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3.,
        # 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 0.,
        # 1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 1., 2., 3., 4., 5., 6., 7.,
        # 8., 9., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
        # [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
        # 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3.,
        # 3., 3., 3., 3., 3., 3., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 5.,
        # 5., 5., 5., 5., 5., 5., 5., 5., 5., 6., 6., 6., 6., 6., 6., 6., 6.,
        # 6., 6., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7.]

        print('xy is',xy)
        xy = gather_operation(xy, max_ind.view(n, -1).int())

        print('xy is',xy)
        # in loop 1, torch.Size([1, 2, 20])        
        # tensor([[[0., 2., 5., 7., 9., 1., 2., 5., 7., 8., 1., 2., 4., 6., 8., 1., 3., 5., 7., 9.],
        #          [0., 1., 1., 0., 0., 2., 3., 2., 3., 3., 4., 4., 5., 5., 4., 7., 6., 6., 6., 7.]]], device='cuda:0')
        xx = xy[:, 0, :].view(n, 1, h//2**i, w//2**i)
        yy = xy[:, 1, :].view(n, 1, h//2**i, w//2**i)
        print('xx is',xx)
        print('yy is',yy)
        mask = (depth > 0).int()
        print('mask is',mask)
        new_mask = torch.zeros_like(mask.view(n, -1))
        print('new_mask is',new_mask)
        # sampling
        vp_num = torch.sum(mask, (2,3)).min()
        print('vp_num is',vp_num)
        num_sam = nsamples[i-1] 
        print('num_sam is',num_sam)
        # notice, only 1 batch
        for j in range(n):
            print('in the only iteration')
            all_idx = torch.arange(mask.shape[2]*mask.shape[3]).reshape(1, -1).cuda().int()

            print('all_idx is',all_idx)
            v_idx = all_idx[mask[j].reshape(1, -1)>0].reshape(1,-1)
            print('v_idx is',v_idx)
            sample = torch.randperm(mask[j].sum())
            print('sample is',sample)
            print('num_sam is',num_sam)
            print('vp_num is',vp_num)
            if vp_num < num_sam:
                v_idx = v_idx[:, sample[:int(vp_num)]]
            else:
                v_idx = v_idx[:, sample[:int(num_sam)]]
            print('v_idx is',v_idx)
            if j == 0:
                s_idx = v_idx
            else:
                s_idx = torch.cat([s_idx, v_idx], 0)
            print('s_idx is',s_idx)

        # gather and 3d points
        # this is (x, y, d)
        xyd = torch.cat((xx, yy, depth), 1).view(n, 3, -1)
        print('xyd is',xyd)
        s_pts = gather_operation(xyd, s_idx).permute(0, 2, 1) 
        print('s_pts is',s_pts)
        cxy = torch.zeros(n,1,3).float().to(depth.get_device())
        print('cxy is',cxy)
        fxy = torch.ones(n,1,3).float().to(depth.get_device())
        print('fxy is',fxy)
        print('K is',K)
        cxy[:,0,0] = K[:,0,2]
        print('cxy is',cxy)
        cxy[:,0,1] = K[:,1,2]
        print('cxy is',cxy)
        fxy[:,0,0] = K[:,0,0]
        print('fxy is',fxy)
        fxy[:,0,1] = K[:,1,1]
        print('fxy is',fxy)
        s_p3d = (s_pts - cxy) / fxy
        print('s_p3d is',s_p3d)
        s_p3d[:,:,0:2] = s_p3d[:,:,0:2] * s_pts[:,:,2:]
        print('s_p3d is',s_p3d)
        # knn
        #nnidx = knn_operation(s_p3d, s_p3d, knn[i-1])
        r=torch.sum(s_p3d*s_p3d, dim=2, keepdim=True)
        print('r is',r)
        m=torch.matmul(s_p3d, s_p3d.transpose(2,1))
        print('m is',m)
        d = r-2*m + r.transpose(2,1)
        print('d is',d)
        _, nnidx=torch.topk(d, k=knn[i-1], dim=-1, largest=False)
        print('nnidx is',nnidx)
        nnidx = nnidx.int()
        print('nnidx is',nnidx)

        spoints.append(s_p3d.permute(0, 2, 1))
        print('spoints is',spoints)
        sidxs.append(s_idx)
        print('sidxs is',sidxs)
        nnidxs.append(nnidx)
        print('nnidxs is',nnidxs)
        new_mask = new_mask.view(n, 1, h//2**i, w//2**i)
        print('new_mask is',new_mask)
        masks.append(new_mask)
        print('masks is',masks)
            
    print('spoints is',spoints)
    print('sidxs is',sidxs)
    print('nnidxs is',nnidxs)
    print('masks is',masks)

if __name__=="__main__":
    test()