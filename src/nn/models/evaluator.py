import torch
from torchvision.utils import make_grid

from src.nn.metrics.evaluation import MultiScore
from src.nn.utils.utils import AverageMeter, AverageMeterDict
from src.nn.utils.mask import generate_segm_class_colors, apply_mask_on_batch, decode_mask_batch


class SegmentationEvaluator:
    def __init__(self, num_classes, device, summary=True, summary_writer=None, class_colors=None):
        self.device = device

        self.num_classes = num_classes
        self.summary = summary
        self.summary_writer = summary_writer

        if class_colors is not None:
            self.class_colors = class_colors
        else:
            self.class_colors = generate_segm_class_colors(self.num_classes)

        self.multi_scores = MultiScore(num_classes=num_classes)

    def evaluate(self, model, dataloader, criterions, epoch):
        print('Evaluation...')
        n_batches = len(dataloader)

        epoch_multi_loss_scalar = AverageMeter()
        epoch_loss_scalars = AverageMeterDict(criterions.losses.keys())
        epoch_score = AverageMeterDict(self.multi_scores.names)

        model.eval()
        with torch.no_grad():
            for batch_index, data in enumerate(dataloader):
                images, targets = data
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = model(images)

                # COMPUTE AND UPDATE LOSSES
                multi_loss_value, single_loss_values = criterions(outputs, targets)

                epoch_multi_loss_scalar.update(multi_loss_value.cpu().item())
                epoch_loss_scalars.avg_update(single_loss_values)

                # COMPUTE AND UPDATE SCORES
                batch_score = self.multi_scores(outputs, targets)
                epoch_score.avg_update(batch_score)

                # WRITE SUMMARIES
                self._write_dict_summary(epoch_loss_scalars, epoch, mode='separate_loss',
                                         n_batches=n_batches, batch_index=batch_index)
                self._write_summary(epoch=epoch, mode='batch_loss', score=multi_loss_value,
                                    n_batches=n_batches, batch_index=batch_index)
                self._write_dict_summary(batch_score, epoch, mode='batch_score',
                                         n_batches=n_batches, batch_index=batch_index)

                if batch_index == (n_batches - 1):
                    self._write_summary(epoch=epoch, mode='end_eval', images=images,
                                        targets=targets, outputs=outputs, scores=epoch_score,
                                        dataloader=dataloader)

        return epoch_score, epoch_multi_loss_scalar, epoch_loss_scalars

    def _write_dict_summary(self, summary_dict, epoch, mode, n_batches, batch_index):

        for key, value in summary_dict.items():
            self._write_summary(epoch=epoch, mode=mode, score=value,
                                n_batches=n_batches, batch_index=batch_index, key=key)

    def _write_summary(self, epoch, mode, images=None, targets=None, outputs=None, scores=None,
                       score=None, n_batches=None, batch_index=None, key=None, dataloader=None):
        """"""
        if self.summary:

            if mode == 'batch_loss':
                self.summary_writer.add_scalar('val/batch/loss', score, epoch * n_batches + batch_index)

            elif mode == 'separate_loss':
                self.summary_writer.add_scalar('train/batch/{}_loss'.format(key), score, epoch * n_batches + batch_index)

            elif mode == 'batch_score':
                self.summary_writer.add_scalar('val/batch/{}'.format(key), score, epoch * n_batches + batch_index)

            elif mode =='end_eval':
                masked_images = apply_mask_on_batch(images, targets, outputs, dataloader)
                targets = decode_mask_batch(targets, dataloader)

                self.summary_writer.add_image('val/image', make_grid(images, normalize=True), epoch)
                self.summary_writer.add_image('val/y_true', make_grid(targets, normalize=True), epoch)
                self.summary_writer.add_image('val/y_pred', make_grid(masked_images, normalize=True), epoch)

                for key, value in scores.items():
                    self.summary_writer.add_scalar('val/epoch/{}'.format(key), value.avg, epoch)

