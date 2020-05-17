import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        """
        :param output: a tensor of predicted masks with shape=(NxCxHxW).
        :param target: a tensor of groudtruth masks with shape=(NxHxW).
        :return:
        """
        output = torch.sigmoid(output)
        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target) + 1e-7
        return 1 - 2 * intersection / union


class JaccardLoss(_Loss):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, output, target):
        """
        :param output: a tensor of predicted masks with shape=(NxCxHxW).
        :param target: a tensor of groudtruth masks with shape=(NxHxW).
        :return:
        """
        eps = 1e-15
        jaccard_target = (target == 1).float()
        jaccard_output = F.sigmoid(output)

        intersection = (jaccard_output * jaccard_target).sum()
        union = jaccard_output.sum() + jaccard_target.sum()

        loss = 1 - (intersection + eps) / (union - intersection + eps)

        return loss


class SmoothJaccardLoss(_Loss):
    def __init__(self, smooth=100):
        super(SmoothJaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        """
        :param output: a tensor of predicted masks with shape=(NxCxHxW).
        :param target: a tensor of groudtruth masks with shape=(NxHxW).
        :return:
        """
        output = torch.sigmoid(output)
        target = target.float()

        intersection = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)

        jac = (intersection + self.smooth) / (union - intersection + self.smooth)
        return 1 - jac


class FocalLossBinary(_Loss):
    """Focal loss puts more weight on more complicated examples.
    https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/losses.py
    output is log_softmax
    """
    def __init__(self, gamma=2, size_average=True, reduce=True):
        super(FocalLossBinary, self).__init__(size_average=size_average, reduce=reduce)
        self.gamma = gamma

    def forward(self, outputs: Tensor, targets: Tensor):

        outputs = F.logsigmoid(outputs)
        logpt = -F.binary_cross_entropy_with_logits(outputs, targets.float(), reduce=False)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt).pow(self.gamma)) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalLossMulti(_Loss):
    """Focal loss puts more weight on more complicated examples.
    https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/losses.py
    output is log_softmax
    """

    def __init__(self, gamma=2, size_average=True, reduce=True, ignore_index=-100, from_logits=False):
        super(FocalLossMulti, self).__init__(size_average=size_average, reduce=reduce)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def forward(self, outputs: Tensor, targets: Tensor):

        if not self.from_logits:
            outputs = F.log_softmax(outputs, dim=1)

        logpt = -F.nll_loss(outputs, targets, ignore_index=self.ignore_index, reduce=False)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt).pow(self.gamma)) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class JaccardLossMulti(_Loss):
    """
    Multiclass jaccard loss
    """
    def __init__(self, ignore_index=-100, from_logits=False, weight: Tensor = None, reduce=True):
        super(JaccardLossMulti, self).__init__(reduce=reduce)
        self.ignore_index = ignore_index
        self.from_logits = from_logits
        self.reduce = reduce

        if weight is None:
            self.class_weights = None
        else:
            self.class_weights = weight / weight.sum()

    def forward(self, outputs: Tensor, targets: Tensor):
        """
        :param outputs: NxCxHxW.
        :param targets: NxHxW.
        :return: scalar
        """
        if self.from_logits:
            outputs = outputs.exp()
        else:
            outputs = torch.softmax(outputs, dim=1)

        eps = 1e-15
        n_classes = outputs.size(1)
        mask = (targets != self.ignore_index)
        smooth = 100
        loss = torch.zeros(n_classes, dtype=torch.float).to(outputs.device)

        for cls_indx in range(0, n_classes):
            jaccard_target = (targets == cls_indx)
            jaccard_output = outputs[:, cls_indx]

            jaccard_target = torch.masked_select(jaccard_target, mask)
            jaccard_output = torch.masked_select(jaccard_output, mask)

            num_preds = jaccard_target.long().sum()

            if num_preds == 0:
                loss[cls_indx] = 0
            else:
                jaccard_target = jaccard_target.float()
                intersection = (jaccard_output * jaccard_target).sum()
                union = jaccard_output.sum() + jaccard_target.sum()
                jac = (intersection + smooth) / (union - intersection + smooth)
                loss[cls_indx] = (1 - jac) / n_classes

        if self.class_weights is not None:
            loss = loss * self.class_weights.to(outputs.device)

        if self.reduce:
            loss = loss.sum()

        return loss

