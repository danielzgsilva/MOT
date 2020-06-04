import torch
from torch.utils.data import DataLoader

import numpy as np
import time
import os
from progress.bar import Bar, ChargingBar

from logger import Logger

from utils.generic_utils import get_dataset, get_model, get_losses, get_optimizer
from utils.generic_utils import update_dataset_and_head_info, load_model, save_model, AverageMeter

from utils.post_process import generic_post_process
from utils.decode import generic_decode, extract_objects
from utils.debugger import Debugger

from networks.resnet_simclr import ResNetSimCLR

from utils.transforms import augment_objects

import cv2

class UnSupervisedTrainer:
    def __init__(self, opt):
        self.data_dir = opt.data_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_dir = opt.save_dir
        self.model_name = opt.model_name
        self.model_arch = opt.model_arch

        # Training parameters
        self.batch_size = int(opt.batch_size)
        self.num_workers = int(opt.num_workers)
        self.epochs = int(opt.num_epochs)
        self.lr = float(opt.lr)
        self.lr_step = opt.lr_step

        DatasetClass = get_dataset(opt.dataset)
        self.opt = update_dataset_and_head_info(opt, DatasetClass)

        self.logger = Logger(self.opt)

        # Create model and place on gpu
        self.model = get_model(self.model_arch, self.opt)
        self.feature_extractor = ResNetSimCLR('resnet18', 256)

        self.start_epoch = 0
        if self.opt.load_model != '':
            self.model, self.optimizer, self.start_epoch = load_model(
                self.model, self.opt.load_model, self.opt, self.optimizer)

        self.model.to(self.device)
        self.feature_extractor.to(self.device)

        # Get optimizer
        self.optimizer = get_optimizer(self.opt.optim, self.lr, self.model)

        # Loss function and optimizer
        self.loss_states, self.loss = get_losses(self.opt)

        print('Training options:\n'
              '\tModel: {}\n\tInput size: {}\n\tBatch size: {}\n'
              '\tEpochs: {}\n\t''Learning rate: {}\n\tLR Steps: {}\n'
              '\tHeads: {}\n\t''Weights: {}\n\tHead Conv: {}\n'
              '\tLosses: {}\n\tDebug level: {}\n'.
              format(self.model_arch.capitalize(), self.opt.input_size, self.batch_size, self.epochs,
                     self.lr, self.lr_step, self.opt.heads, self.opt.weights, self.opt.head_conv,
                     self.loss_states, self.opt.debug))

        # Creating PyTorch datasets
        self.datasets = dict()
        self.datasets['train'] = DatasetClass(self.opt, 'train')
        self.datasets['val'] = DatasetClass(self.opt, 'val')

        self.dataset_lens = [self.datasets[i].__len__() for i in ['train', 'val']]

        print('Training on:\n\tTrain files: {}\n\tValidation files: {}\n'.format(*self.dataset_lens))

        # Creating PyTorch dataloaders
        self.dataloaders = {i: DataLoader(self.datasets[i], batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers, pin_memory=True, drop_last=True)
                            for i in ['train', 'val']}

    def run_epoch(self, phase, epoch):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()

        avg_loss_stats = {l: AverageMeter() for l in self.loss_states \
                          if l == 'tot' or self.opt.weights[l] > 0}

        bar = Bar('{}'.format(self.model_name), max=len(self.dataloaders[phase]))
        end = time.time()

        # batch keys: image', 'pre_img', 'pre_hm', 'hm', 'ind', 'cat', 'mask', 'reg', 'reg_mask', 'wh', 'wh_mask', 'tracking', 'tracking_mask
        for i, batch in enumerate(self.dataloaders[phase]):
            data_time.update(time.time() - end)

            # Ensure we're doing this calculation on our GPU if possible
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=self.device, non_blocking=True)

            pre_img = batch['pre_img'] if 'pre_img' in batch else None
            pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None

            # Calculate gradients only if we're in the training phase
            with torch.set_grad_enabled(phase == 'train'):

                # output keys: hm, tracking, reg, wh
                outputs = self.model(batch['image'], pre_img, pre_hm)

                # Calculate the loss of the batch
                loss, loss_states = self.loss(outputs, batch)
                loss = loss.mean()

                # Figure out why generic decode has to be done after loss computation
                # Extract detected objects
                objs = extract_objects(batch, outputs[-1], self.opt)

                # Apply transformations to objects
                xis, xjs = augment_objects(objs)

                for b in range(len(xis)):
                    for xi, xj in zip(xis[b], xjs[b]):
                        xi = xi.detach().cpu().numpy().transpose(1, 2, 0)
                        xi = np.clip(((xi * self.datasets['train'].std + self.datasets['train'].mean) * 255.), 0, 255).astype(np.uint8)
                        xj = xj.detach().cpu().numpy().transpose(1, 2, 0)
                        xi = np.clip(((xj * self.datasets['train'].std + self.datasets['train'].mean) * 255.), 0,
                                     255).astype(np.uint8)
                        cv2.imshow('xi', xi)
                        cv2.imshow('xj', xj)

                # Generate feature vectors

                # Get ground truth objs from previous image
                # Find way to visualize objs -> augmented -> clustering
                # Get w leul for bipartite grpah and clustering

                # Adjust weights through backprop if we're in training phase
                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]| Tot: {total:} | ETA: {eta:} '.format(
                epoch, bar.index, bar.max, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_states[l].mean().item(), batch['image'].size(0))
                Bar.suffix = Bar.suffix + ' | {}: {:.4f} '.format(l, avg_loss_stats[l].avg)

            Bar.suffix = Bar.suffix + '| Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                      '| Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

            bar.next()

            if self.opt.debug > 0:
                self.debug(batch, outputs[-1], i, dataset=self.dataloaders[phase].dataset)

            del outputs, loss, loss_states

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def train(self):
        start = time.time()

        # best_model_wts = self.model.state_dict()
        # best_acc = 0.0

        # print('| Epoch\t | Train Loss\t| Train Acc\t| Valid Loss\t| Valid Acc\t| Epoch Time |')
        # print('-' * 86)

        # Iterate through epochs
        for epoch in range(1, self.epochs + 1):

            epoch_start = time.time()
            self.logger.write('epoch: {} | '.format(epoch))

            # Training phase
            log_dict_train, _ = self.run_epoch('train', epoch)

            for k, v in log_dict_train.items():
                self.logger.scalar_summary('train_{}'.format(k), v, epoch)
                self.logger.write('{} {:7f} | '.format(k, v))

            # Validation phase
            log_dict_val, preds = self.run_epoch('val', epoch)

            if self.opt.eval_val:
                self.dataloaders['val'].dataset.run_eval(preds, self.opt.save_dir)

            for k, v in log_dict_val.items():
                self.logger.scalar_summary('val_{}'.format(k), v, epoch)
                self.logger.write('{} {:7f} | '.format(k, v))

            self.logger.write('\n')
            epoch_time = time.time() - epoch_start

            ''' # Print statistics after the validation phase
            print("| {}\t | {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.0f}m {:.0f}s     |"
                  .format(epoch, train_loss, train_acc, val_loss, val_acc,
                          epoch_time // 60, epoch_time % 60))'''

            # Copy and save the model's weights if it has the best accuracy thus far
            '''if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = self.model.state_dict()'''

            # Possibly save model
            if self.opt.save_period > 0 and epoch % self.opt.save_period == 0:
                save_model(os.path.join(self.opt.model_folder, '{}_ep{}.pth'.format(self.model_name, epoch)),
                           epoch, self.model, self.optimizer)

            # Possibly update learning rate
            if epoch in self.opt.lr_step:
                lr = self.opt.lr * (0.1 ** (self.opt.lr_step.index(epoch) + 1))
                print('Drop LR to', lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

        total_time = time.time() - start

        # print('-' * 86)
        print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
        # print('Best validation accuracy: {:.4f}'.format(best_acc))

        # load best model weights and save them
        # self.model.load_state_dict(best_model_wts)

        save_model(os.path.join(self.opt.model_folder, '{}_{}.pth'.format(self.model_name, 'best')),
                   self.opt.num_epochs, self.model, self.optimizer)

        self.logger.close()
        return

    def debug(self, batch, output, iter_id, dataset):
        opt = self.opt
        if 'pre_hm' in batch:
            output.update({'pre_hm': batch['pre_hm']})

        dets = generic_decode(output, K=opt.K, opt=opt)
        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()

        dets_gt = batch['meta']['gt_det']

        for i in range(1):
            debugger = Debugger(opt=opt, dataset=dataset)
            img = batch['image'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')

            if 'pre_img' in batch:
                pre_img = batch['pre_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
                pre_img = np.clip(((pre_img * dataset.std + dataset.mean) * 255), 0, 255).astype(np.uint8)
                debugger.add_img(pre_img, 'pre_img_pred')
                debugger.add_img(pre_img, 'pre_img_gt')
                if 'pre_hm' in batch:
                    pre_hm = debugger.gen_colormap(
                        batch['pre_hm'][i].detach().cpu().numpy())
                    debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

            debugger.add_img(img, img_id='out_pred')
            if 'ltrb_amodal' in opt.heads:
                debugger.add_img(img, img_id='out_pred_amodal')
                debugger.add_img(img, img_id='out_gt_amodal')

            # Predictions
            for k in range(len(dets['scores'][i])):
                if dets['scores'][i, k] > opt.vis_thresh:
                    debugger.add_coco_bbox(
                        dets['bboxes'][i, k] * opt.down_ratio, dets['clses'][i, k],
                        dets['scores'][i, k], img_id='out_pred')

                    if 'ltrb_amodal' in opt.heads:
                        debugger.add_coco_bbox(
                            dets['bboxes_amodal'][i, k] * opt.down_ratio, dets['clses'][i, k],
                            dets['scores'][i, k], img_id='out_pred_amodal')

                    if 'hps' in opt.heads and int(dets['clses'][i, k]) == 0:
                        debugger.add_coco_hp(
                            dets['hps'][i, k] * opt.down_ratio, img_id='out_pred')

                    if 'tracking' in opt.heads:
                        debugger.add_arrow(
                            dets['cts'][i][k] * opt.down_ratio,
                            dets['tracking'][i][k] * opt.down_ratio, img_id='out_pred')
                        debugger.add_arrow(
                            dets['cts'][i][k] * opt.down_ratio,
                            dets['tracking'][i][k] * opt.down_ratio, img_id='pre_img_pred')

                    # Extracting detected object
                    x1, y1, x2, y2 = [int(round(x)) for x in dets['bboxes'][i, k] * opt.down_ratio]
                    obj = img[y1:y2, x1:x2, :]
                    name = '{}{:.1f}'.format(debugger.names[int(dets['clses'][i, k])], dets['scores'][i, k])
                    debugger.add_img(obj, img_id=name)

            # Ground truth
            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt['scores'][i])):
                if dets_gt['scores'][i][k] > opt.vis_thresh:
                    debugger.add_coco_bbox(
                        dets_gt['bboxes'][i][k] * opt.down_ratio, dets_gt['clses'][i][k],
                        dets_gt['scores'][i][k], img_id='out_gt')

                    if 'ltrb_amodal' in opt.heads:
                        debugger.add_coco_bbox(
                            dets_gt['bboxes_amodal'][i, k] * opt.down_ratio,
                            dets_gt['clses'][i, k],
                            dets_gt['scores'][i, k], img_id='out_gt_amodal')

                    if 'hps' in opt.heads and \
                            (int(dets['clses'][i, k]) == 0):
                        debugger.add_coco_hp(
                            dets_gt['hps'][i][k] * opt.down_ratio, img_id='out_gt')

                    if 'tracking' in opt.heads:
                        debugger.add_arrow(
                            dets_gt['cts'][i][k] * opt.down_ratio,
                            dets_gt['tracking'][i][k] * opt.down_ratio, img_id='out_gt')
                        debugger.add_arrow(
                            dets_gt['cts'][i][k] * opt.down_ratio,
                            dets_gt['tracking'][i][k] * opt.down_ratio, img_id='pre_img_gt')

            if 'hm_hp' in opt.heads:
                pred = debugger.gen_colormap_hp(
                    output['hm_hp'][i].detach().cpu().numpy())
                gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
                debugger.add_blend_img(img, pred, 'pred_hmhp')
                debugger.add_blend_img(img, gt, 'gt_hmhp')

            if 'rot' in opt.heads and 'dim' in opt.heads and 'dep' in opt.heads:
                dets_gt = {k: dets_gt[k].cpu().numpy() for k in dets_gt}
                calib = batch['meta']['calib'].detach().numpy() \
                    if 'calib' in batch['meta'] else None
                det_pred = generic_post_process(opt, dets,
                                                batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
                                                output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
                                                calib)
                det_gt = generic_post_process(opt, dets_gt,
                                              batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
                                              output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
                                              calib)

                debugger.add_3d_detection(
                    batch['meta']['img_path'][i], batch['meta']['flipped'][i],
                    det_pred[i], calib[i],
                    vis_thresh=opt.vis_thresh, img_id='add_pred')
                debugger.add_3d_detection(
                    batch['meta']['img_path'][i], batch['meta']['flipped'][i],
                    det_gt[i], calib[i],
                    vis_thresh=opt.vis_thresh, img_id='add_gt')
                debugger.add_bird_views(det_pred[i], det_gt[i],
                                        vis_thresh=opt.vis_thresh, img_id='bird_pred_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)
