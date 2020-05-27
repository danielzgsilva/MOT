import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time
import os
from progress.bar import Bar

from .logger import Logger
from .utils.utils import get_model, get_optimizer, get_dataset, get_losses, \
    load_model, save_model, update_dataset_and_head_info, AverageMeter


class UnSupervisedTrainer:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_dir = args.save_dir
        self.model_name = args.model_name
        self.model_arch = args.model_arch

        # Training parameters
        self.batch_size = int(args.batch_size)
        self.num_workers = int(args.num_workers)
        self.epochs = int(args.num_epochs)
        self.lr = float(args.learning_rate)
        self.lr_step = args.lr_step

        DatasetClass = get_dataset(args.dataset)
        self.opt = update_dataset_and_head_info(args, DatasetClass)

        self.logger = Logger(self.opt)

        # Create model and place on gpu
        self.model = get_model(self.model_arch, self.opt)

        self.start_epoch = 0
        if self.opt.load_model != '':
            self.model, self.optimizer, self.start_epoch = load_model(
                self.model, self.opt.load_model, self.opt, self.optimizer)

        self.model.to(self.device)
        print(self.model)

        # Get optimizer
        self.optimizer = get_optimizer(self.opt.optim, self.lr, self.model)

        # Loss function and optimizer
        self.loss_stats, self.loss = get_losses(self.opt)

        print('Training options:\n'
              '\tModel: {}\n\tInput size: {}\n\tBatch size: {}\n'
              '\tEpochs: {}\n\t''Learning rate: {}\n\tLR Steps: {}\n'.
              format(self.model_arch.capitalize(), self.opt.input_size, self.batch_size, self.epochs,
                     self.lr, self.lr_step))

        # Creating PyTorch datasets
        self.datasets = dict()
        self.datasets['train'] = DatasetClass(self.opt, 'train')
        self.datasets['val'] = DatasetClass(self.opt, 'val')

        self.dataset_lens = [self.datasets[i].__len__() for i in ['train', 'val']]

        print('Training on:\n'
              '\tTrain files: {}\n\tValidation files: {}\n'
              .format(*self.dataset_lens))

        # Creating PyTorch dataloaders
        self.dataloaders = {i: DataLoader(self.datasets[i], batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers, pin_memory=True, drop_last=True)
                            for i in ['train', 'val']}

    def run_epoch(self, phase, epoch):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        data_time, batch_time = AverageMeter(), AverageMeter()

        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats \
                          if l == 'tot' or self.opt.weights[l] > 0}

        num_iters = len(self.dataloaders[phase])
        bar = Bar('{}'.format(self.model_name, max=num_iters))
        end = time.time()

        # Looping through batches
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

                # This calls the forward() function on a batch of inputs
                outputs = self.model(batch['image'], pre_img, pre_hm)

                # Calculate the loss of the batch
                loss, loss_stats = self.loss(outputs, batch)
                loss = loss.mean()

                # Adjust weights through backprop if we're in training phase
                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, i, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['image'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                      '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

            # print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            bar.next()

            del outputs, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret

    def train(self):
        start = time.time()

        # best_model_wts = self.model.state_dict()
        # best_acc = 0.0

        # print('| Epoch\t | Train Loss\t| Train Acc\t| Valid Loss\t| Valid Acc\t| Epoch Time |')
        # print('-' * 86)

        # Iterate through epochs
        for epoch in range(1, self.epochs + 1):

            epoch_start = time.time()
            self.logger.write('epoch: {} |'.format(epoch))

            # Training phase
            log_dict_train = self.run_epoch('train', epoch)

            for k, v in log_dict_train.items():
                self.logger.scalar_summary('train_{}'.format(k), v, epoch)
                self.logger.write('{} {:8f} | '.format(k, v))

            # Validation phase
            log_dict_val = self.run_epoch('val', epoch)

            # if opt.eval_val:
            #    val_loader.dataset.run_eval(preds, opt.save_dir)

            for k, v in log_dict_val.items():
                self.logger.scalar_summary('val_{}'.format(k), v, epoch)
                self.logger.write('{} {:8f} | '.format(k, v))

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
