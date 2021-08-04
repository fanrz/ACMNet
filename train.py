import time
import torch.nn
from options.train_options import TrainOptions
from data import create_dataloader
from models import create_model
from util import SaveResults
import numpy as np
import cv2

if __name__ == '__main__':
    opt = TrainOptions().parse()
    print('opt will be printed.')
    print(opt)
    # batchSize=2, 
    # beta1=0.9, channels=64, checkpoints_dir='./checkpoints', clip=True, 
    # continue_train=False, dataset='kitti', epoch_count=1, expr_name='kitti_dcomp', 
    # gpu_ids=[0], init_gain=0.02, init_type='kaiming', isTrain=True, knn=[6, 6, 6], 
    # lambda_R=1.0, lambda_S=0.01, lr=0.0005, lr_decay_iters=10, lr_policy='step', 
    # model='dcomp', nThreads=8, niter=50, no_augment=False, no_flip=False, nsamples=[10000, 5000, 2500], 
    # optimizer='adam', print_freq=32, root='datasets', save_epoch_freq=1, 
    # save_latest_freq=6400, save_result_freq=3200, scale=80, suffix='', test_data_file='sval.list', 
    # train_data_file='train.list', which_epoch='latest'
    print('model will be printed out')

    train_data_loader = create_dataloader(opt)
    train_dataset_size = len(train_data_loader) 
    print('#'*20)
    print('#training images = %d' % train_dataset_size)

    model = create_model(opt)
    print(model)

    # train_data_loader = create_dataloader(opt)
    # train_dataset_size = len(train_data_loader)   
    # print('#training images = %d' % train_dataset_size)

    model.setup(opt)
    save_results = SaveResults(opt)
    total_steps = 0

    lr = opt.lr
    print('opt.lr is',opt.lr)

    for epoch in range(opt.epoch_count, opt.niter + 1):
        print('opt.epoch_count is',opt.epoch_count)
        print('opt.niter is',opt.niter)
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # training
        print("training stage (epoch: %s) starting...................." % epoch)
        for ind, data in enumerate(train_data_loader):
            print('this is ind',ind)
            # here, data is [batch, ]
            iter_start_time = time.time()
            print('iter_start_time is ', iter_start_time)
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                save_results.print_current_losses(epoch, epoch_iter, lr, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, total_steps))
                model.save_networks('latest')

            if total_steps % opt.save_result_freq == 0:
                save_results.save_current_results(model.get_current_visuals(), epoch)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter, time.time() - epoch_start_time))
        lr = model.update_learning_rate()
