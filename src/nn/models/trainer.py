import numpy as np
import pandas as pd
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from src.nn.models.evaluator import SegmentationEvaluator
from src.nn.utils.mask import decode_mask_batch, apply_mask_on_batch
from src.nn.utils.utils import save_snapshot, restore_snapshot, load_checkpoint_enet, AverageMeter, AverageMeterDict


# TODO: add regularization terms
class SegmentationTrainer:
    def __init__(self, model, train_loader, test_loader, optimizer, multi_loss, device,
                 log_interval, num_epoch, log_dir, checkpoint_path, pretrained,
                 image_shape, scale, summary=True, scheduler=None):
        self.model = model
        self.pretrained = pretrained
        self.global_training_step = 0

        self.image_shape = image_shape

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.device = device

        self.num_epoch = num_epoch

        self.multi_loss = multi_loss
        self.optimizer = optimizer

        self.scheduler = scheduler

        num_classes = model.num_classes

        self.summary = summary

        self.summary_writer = SummaryWriter(log_dir=log_dir)
        self.evaluator = SegmentationEvaluator(num_classes, device=self.device,
                                               summary=self.summary, summary_writer=self.summary_writer)

        self.checkpoint_path = checkpoint_path
        self.log_interval = log_interval

    def train(self):
        """
        """

        if self.pretrained:
            # TODO: change for other models
            if self.model.name == '...':
                self.model = load_checkpoint_enet(self.model, self.checkpoint_path)
                # TODO: rewrite it or move into SegmentationModel class
                self.model.basemodel.transposed_conv = nn.ConvTranspose2d(
                    16, self.model.num_classes, kernel_size=3, stride=2, padding=1, bias=False).to(self.device)
                start_epoch = 0
                best_loss = np.inf
                train_history = pd.DataFrame()

            else:
                self.model, start_epoch, train_history, best_loss = restore_snapshot(self.model, self.optimizer, self.checkpoint_path)

        else:
            start_epoch = 0
            best_loss = np.inf
            train_history = pd.DataFrame()

        for epoch in range(start_epoch, self.num_epoch):
            print('----------------------------------------')
            print('EPOCH = ', epoch)

            # TRAIN MODEL ONE EPOCH
            train_scores, train_losses, _ = self._train_epoch(epoch)
            print('train_losses = ', train_losses)

            # TEST MODEL
            if (epoch % 1 == 0) or ((epoch-1) == self.num_epoch):
                val_score_scalars, val_epoch_loss_scalar, _ = self.evaluator.evaluate(self.model, self.test_loader,
                                                                                      self.multi_loss, epoch)
                print('test_losses = ', val_epoch_loss_scalar)

                if self.scheduler is not None:
                    self.scheduler.step(val_epoch_loss_scalar.avg)

                summary = {
                    'epoch': [epoch],
                    'loss': [train_losses.avg],
                    'val_loss': [val_epoch_loss_scalar.avg]
                }
                train_history = train_history.append(pd.DataFrame.from_dict(summary), ignore_index=True)

                # SAVE THE BEST PARAMS OF MODEL
                if val_epoch_loss_scalar.avg < best_loss:
                    save_snapshot(self.model, self.optimizer, val_epoch_loss_scalar.avg,
                                  epoch, train_history, self.checkpoint_path)
                    best_loss = val_epoch_loss_scalar.avg
                    print('Checkpoint saved', epoch, best_loss)

                if (epoch-1) == self.num_epoch:
                    last_checkpoint_path = self.checkpoint_path.split('.')[0] + '-epoch_{}.{}'.format(
                        epoch, self.checkpoint_path.split('.')[1])
                    save_snapshot(self.model, self.optimizer, val_epoch_loss_scalar.avg,
                                  self.num_epoch, train_history, last_checkpoint_path)

    def _train_epoch(self, epoch):
        """"""
        n_batches = len(self.train_loader)

        # INITIALIZE AVERAGE DICTS FOR LOSSES AND SCORES
        epoch_multi_loss_scalar = AverageMeter()
        epoch_loss_scalars = AverageMeterDict(self.multi_loss.losses.keys())
        epoch_score_values = AverageMeterDict(self.evaluator.multi_scores.names)

        self.model.train()
        for batch_index, data in enumerate(self.train_loader):
            images, targets = data
            images, targets = images.to(self.device), targets.to(self.device)
        
            self.optimizer.zero_grad()

            outputs = self.model(images)
            # COMPUTE LOSS + COMPUTE GRADIENTS + OPTIMIZER STEP
            multi_loss_value, single_loss_values = self.multi_loss(outputs, targets)
            multi_loss_value.backward()
            self.optimizer.step()

            epoch_multi_loss_scalar.update(multi_loss_value.cpu().item())
            epoch_loss_scalars.avg_update(single_loss_values)

            self.global_training_step += 1

            # COMPUTE SCORES
            batch_score_values = self.evaluator.multi_scores(outputs, targets)
            epoch_score_values.avg_update(batch_score_values)

            # WRITE SUMMARIES
            self._write_dict_summary(single_loss_values, epoch, mode='separate_loss',
                                     n_batches=n_batches, batch_index=batch_index)
            self._write_summary(epoch=epoch, mode='batch_end', score=multi_loss_value.cpu().item(),
                                n_batches=n_batches, batch_index=batch_index)
            self._write_dict_summary(batch_score_values, epoch, mode='eval_train',
                                     n_batches=n_batches, batch_index=batch_index)

            if batch_index == (n_batches - 1):
                self._write_summary(epoch=epoch, mode="train_end", images=images,
                                    targets=targets, outputs=outputs, scores=epoch_score_values)

        return epoch_score_values, epoch_multi_loss_scalar, epoch_loss_scalars

    def _write_dict_summary(self, summary_dict, epoch, mode, n_batches, batch_index):

        for key, value in summary_dict.items():
            self._write_summary(epoch=epoch, mode=mode, score=value,
                                n_batches=n_batches, batch_index=batch_index, key=key)

    def _write_summary(self, epoch, mode, images=None, targets=None, outputs=None, scores=None,
                       score=None, n_batches=None, batch_index=None, key=None):

        if self.summary:

            if mode == 'scheduler':
                if self.scheduler is not None:
                    self.scheduler.step(epoch)
                    lrs = self.scheduler.get_lr()

                    if len(lrs) > 1:
                        self.summary_writer.add_scalars('train/lr', dict(enumerate(lrs)), global_step=epoch)
                    else:
                        self.summary_writer.add_scalar('train/lr', lrs[0], global_step=epoch)

            elif mode == 'eval_train':
                self.summary_writer.add_scalar('train/batch/{}'.format(key), score, epoch * n_batches + batch_index)

            elif mode == 'separate_loss':
                self.summary_writer.add_scalar('train/batch/{}_loss'.format(key), score, epoch * n_batches + batch_index)

            elif mode == 'batch_end':
                self.summary_writer.add_scalar('train/batch/loss', score, epoch * n_batches + batch_index)

                # Plot gradient absmax and absmin to see if there are any gradient explosions
                grad_max = 0
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_max = max(grad_max, param.grad.abs().max().cpu().item())

                self.summary_writer.add_scalar('train/grad/global_abs_max', grad_max, epoch * n_batches + batch_index)

            elif mode == 'train_end':
                masked_images = apply_mask_on_batch(images, targets, outputs, self.train_loader)
                targets = decode_mask_batch(targets, self.train_loader)

                self.summary_writer.add_image('train/image', make_grid(images, normalize=True), epoch)
                self.summary_writer.add_image('train/y_true', make_grid(targets, normalize=True), epoch)
                self.summary_writer.add_image('train/y_pred', make_grid(masked_images, normalize=True), epoch)

                for key, value in scores.items():
                    self.summary_writer.add_scalar('train/epoch/{}'.format(key), value.avg, epoch)

                # Plot histogram of parameters after each epoch
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        # Plot weighs
                        param_data = param.data.cpu().numpy()
                        self.summary_writer.add_histogram('model/{}'.format(name), param_data, epoch, bins='doane')



