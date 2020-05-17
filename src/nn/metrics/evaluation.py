import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from src.nn.utils.utils import interpolate_mask


class MultiScore(_Loss):
    """
    Metrics aggregation class.
    It was created in order to unify training pipeline.
    """
    def __init__(self, num_classes):
        super(MultiScore, self).__init__()
        self.num_classes = num_classes

        self.scores = dict()

        if self.num_classes > 1:
            self.scores['JaccardScore'] = JaccardScoreMulti()
        else:
            self.scores['JaccardScore'] = JaccardScore()
            self.scores['PixelAcc'] = PixelAccuracyBinary()

        self.names = self.scores.keys()

    def forward(self, output, target):
        """
        :param output: model predictions of size=(n_samples, n_classes, height, width)
                       or each class there is own channel.
        :param target: target labels of size=(n_samples, height, width),
                       for each class there is own number from 0 to (n_classes-1).
        :return:
        """

        # preprocess mask
        if output.size() != target.size():
            output, target = interpolate_mask(output, target, flatten=False)

        if self.num_classes <= 1:
            output = torch.squeeze(output, 1).type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)

        result = dict()
        for name, score in self.scores.items():
            result[name] = score(output, target)

        return result


class JaccardScore(_Loss):
    def __init__(self):
        super(JaccardScore, self).__init__()

    def forward(self, output, target):
        """
        :param output: a tensor of predicted masks with shape=(NxCxHxW).
        :param target: a tensor of groudtruth masks with shape=(NxHxW).
        :return:
        """
        output = torch.sigmoid(output) > 0.5
        target = target.byte()

        intersection = torch.sum(output * target).float()
        union = torch.sum(output) + torch.sum(target)

        union = union.float()
        jac = intersection / (union - intersection + 1e-7)
        return jac

    def __str__(self):
        return 'JaccardScore'


class PixelAccuracyBinary(_Loss):
    def __init__(self):
        super(PixelAccuracyBinary, self).__init__()

    def forward(self, output, target):
        """
        :param output: a tensor of predicted masks with shape=(NxCxHxW).
        :param target: a tensor of groudtruth masks with shape=(NxHxW).
        :return:
        """
        output = F.sigmoid(output) > 0.5
        target = target.byte()

        n_true = torch.eq(output, target)
        n_all = torch.numel(target)
        n_true = n_true.sum()

        if n_true == 0:
            return n_true

        return n_true.float() / n_all

    def __str__(self):
        return 'PixelAccuracy'


class JaccardScoreMulti(_Loss):
    """
    Multi class jaccard score.
    """
    def __init__(self, ignore_index=-100, from_logits=False, weight: Tensor = None, reduce=True):
        super(JaccardScoreMulti, self).__init__(reduce=reduce)
        self.ignore_index = ignore_index
        self.from_logits = from_logits
        self.reduce = reduce

        if weight is None:
            self.class_weights = None
        else:
            self.class_weights = weight / weight.sum()

    def forward(self, outputs: Tensor, targets: Tensor):
        """
        :param outputs: NxCxHxW
        :param targets: NxHxW
        :return: scalar
        """
        if self.from_logits:
            outputs = outputs.exp() > 0.5
        else:
            outputs = torch.softmax(outputs, dim=1) > 0.5

        outputs = outputs.float()

        n_classes = outputs.size(1)
        mask = (targets != self.ignore_index)
        smooth = 100
        score = torch.zeros(n_classes, dtype=torch.float).to(outputs.device)

        for cls_indx in range(0, n_classes):
            jaccard_target = (targets == cls_indx)
            jaccard_output = outputs[:, cls_indx]

            jaccard_target = torch.masked_select(jaccard_target, mask)
            jaccard_output = torch.masked_select(jaccard_output, mask)

            num_preds = jaccard_target.long().sum()

            if num_preds == 0:
                score[cls_indx] = 0
            else:
                jaccard_target = jaccard_target.float()
                intersection = (jaccard_output * jaccard_target).sum()
                union = jaccard_output.sum() + jaccard_target.sum()
                jac = (intersection + smooth) / (union - intersection + smooth)
                score[cls_indx] = jac / n_classes

        if self.class_weights is not None:
            score = score * self.class_weights.to(outputs.device)

        if self.reduce:
            score = score.sum()

        return score

