import torch
from torch.nn.modules.loss import _Loss, BCEWithLogitsLoss, NLLLoss, CrossEntropyLoss

from src.nn.utils.utils import interpolate_mask
from src.nn.metrics.losses import JaccardLoss, FocalLossBinary, DiceLoss, \
    SmoothJaccardLoss, JaccardLossMulti, FocalLossMulti


class MultiLoss(_Loss):
    """
    Loss aggregation class.
    It was created in order to unify training pipeline
    and make possible multitask learning approach.
    """
    def __init__(self, loss_names, loss_weights, device, num_classes):
        """
        :param loss_names: list of loss names,
            possible losses=['jaccard', 'nlll', 'crossentropy', 'smooth_jaccard', 'focal', 'dice']
        :param loss_weights: list of weight coefficients for each loss from loss_names.
        :param device: execution device.
        :param num_classes: number of classes in training data.
        """
        super(MultiLoss, self).__init__()
        assert len(loss_names) == len(loss_weights)
        self.device = device
        self.losses = dict()
        self.loss_weights = dict()
        self.num_classes = num_classes

        for loss, weight in zip(loss_names, loss_weights):
            loss = loss.lower()

            if loss == 'jaccard':
                if self.num_classes > 1:
                    self.losses[loss] = JaccardLossMulti()
                else:
                    self.losses[loss] = JaccardLoss()

            elif loss == 'nlll':
                # if last layer of network is softmax use NLLLoss if isn't use CrossEntropyLoss
                self.losses[loss] = NLLLoss()

            elif loss == 'crossentropy':
                # if last layer of network is softmax use NLLLoss if isn't use CrossEntropyLoss
                # or BCEWithLogitsLoss for binary case
                if self.num_classes > 1:
                    self.losses[loss] = CrossEntropyLoss(size_average=True)
                else:
                    self.losses[loss] = BCEWithLogitsLoss(size_average=True)

            elif loss == 'smooth_jaccard':
                if self.num_classes > 1:
                    raise ValueError("ERROR: for multiclass case loss is not implemented.")
                else:
                    self.losses[loss] = SmoothJaccardLoss()

            elif loss == 'focal':
                if self.num_classes > 1:
                    self.losses[loss] = FocalLossMulti(size_average=False)
                else:
                    self.losses[loss] = FocalLossBinary(size_average=False)

            elif loss == 'dice' and self.num_classes < 2:
                self.losses[loss] = DiceLoss()

            else:
                raise ValueError(loss)

            self.loss_weights[loss] = weight

    def forward(self, output, target):
        """
        :param output: model predictions of size=(n_samples, n_classes, height, width)
            or each class there is own channel.
        :param target: target labels of size=(n_samples, height, width),
            for each class there is own number from 0 to (n_classes-1).
        :return:
        """
        assert len(output) == len(target)

        # preprocess mask
        output, target = interpolate_mask(output, target, flatten=False)

        output = torch.squeeze(output, 1).type(torch.FloatTensor)
        target = target.type(torch.LongTensor)

        single_loss_values = dict()
        for loss_name, loss in self.losses.items():
            single_loss_values[loss_name] = loss(output, target)

        multi_loss = sum([loss * self.loss_weights[loss_name] for (loss_name, loss) in single_loss_values.items()])

        return multi_loss, single_loss_values


