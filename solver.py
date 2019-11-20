# Created on 2018/12
# Author: Kaituo XU

import os
import time
import torch
import numpy
from pit_criterion import cal_loss
from utils import device

class Solver(object):

    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        self.batch_size  = args.batch_size
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            self.model.load_state_dict(torch.load(self.continue_from, map_location='cpu'))
            self.start_epoch = 0
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.state_dict(), file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            if epoch % 2 == 0 and epoch != 0:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] * 0.98
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))

            # # Adjust learning rate (halving)
            # if self.half_lr:
            #     if val_loss >= self.prev_val_loss:
            #         self.val_no_impv += 1
            #         if self.val_no_impv >= 3:
            #             self.halving = True
            #         if self.val_no_impv >= 10 and self.early_stop:
            #             print("No imporvement for 10 epochs, early stopping.")
            #             break
            #     else:
            #         self.val_no_impv = 0
            # if self.halving:
            #     optim_state = self.optimizer.state_dict()
            #     optim_state['param_groups'][0]['lr'] = \
            #         optim_state['param_groups'][0]['lr'] / 2.0
            #     self.optimizer.load_state_dict(optim_state)
            #     print('Learning rate adjusted to: {lr:.6f}'.format(
            #         lr=optim_state['param_groups'][0]['lr']))
            #     self.halving = False
            # self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            best_file_path = os.path.join(
                self.save_folder, 'temp_best.pth.tar')
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_file_path)
                print("Find better validated model, saving to %s" % best_file_path)

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        print('data_loader.len {}'.format(len(data_loader)))
        for i, (data) in enumerate(data_loader):
            padded_mixture_, mixture_lengths_, padded_source_ = data
            seg_idx = numpy.random.randint(0, padded_mixture_.shape[0], self.batch_size)
            padded_mixture = padded_mixture_[seg_idx, :]
            mixture_lengths = mixture_lengths_[seg_idx]
            padded_source = padded_source_[seg_idx, :]
            # print('seg_idx {}'.format(seg_idx))
            # print('padded_mixture_ {}'.format(padded_mixture_))
            # print('padded_source_ {}'.format(padded_source_))
            # print('padded_mixture {}'.format(padded_mixture))
            # print('padded_source {}'.format(padded_source))
            # print('mixture_lengths {}'.format(mixture_lengths))
            # print('padded_mixture.shape {}'.format(padded_mixture.shape))
            if self.use_cuda:
                # padded_mixture = padded_mixture.cuda()
                # mixture_lengths = mixture_lengths.cuda()
                # padded_source = padded_source.cuda()
                padded_mixture = padded_mixture.to(device)
                mixture_lengths = mixture_lengths.to(device)
                padded_source = padded_source.to(device)
            #print('padded_mixture.shape {}'.format(padded_mixture.shape))
            estimate_source = self.model(padded_mixture)
            #print('estimate_source.shape {}'.format(estimate_source.shape))
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                    epoch + 1, i + 1, total_loss / (i + 1),
                    loss.item(), 1000 * (time.time() - start) / (i + 1)),
                    flush=True)

        return total_loss / (i + 1)
