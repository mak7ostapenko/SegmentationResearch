import torch.nn as nn

from src.nn.models.enet.enet import ENet
from src.nn.models.unet.unet import UNet
from src.nn.models.pspnet.pspnet import PSPNet
from src.nn.models.deeplab.deeplab import DeepLab


class SegmentationModel(nn.Module):
    def __init__(self, num_input_channels, num_classes, name='unet',  backbone='resnet18', pretrained=True):
        super(SegmentationModel, self).__init__()

        self.name = name
        self.num_classes = num_classes

        if self.name == 'unet':
            self.basemodel = UNet(in_channels=num_input_channels, n_classes=num_classes)

        elif self.name == "enet":
            if pretrained:
                # TODO: change it
                # WARNING: 12 is num classes in CamVid dataset
                default_num_classes = num_classes #12
                self.basemodel = ENet(in_channels=num_input_channels, num_classes=default_num_classes)
            else:
                self.basemodel = ENet(in_channels=num_input_channels, num_classes=num_classes)

        elif self.name == "icnet":
            pass

        elif self.name == 'pspnet':
            psp_models = {
                'squeezenet': lambda: PSPNet(num_classes=self.num_classes, sizes=(1, 2, 3, 6), psp_size=512,
                                             deep_features_size=256, backend='squeezenet'),
                'densenet': lambda: PSPNet(num_classes=self.num_classes, sizes=(1, 2, 3, 6), psp_size=1024,
                                           deep_features_size=512, backend='densenet'),
                'resnet18': lambda: PSPNet(num_classes=self.num_classes, sizes=(1, 2, 3, 6), psp_size=512,
                                           deep_features_size=256, backend='resnet18'),
                'resnet34': lambda: PSPNet(num_classes=self.num_classes, sizes=(1, 2, 3, 6), psp_size=512,
                                           deep_features_size=256, backend='resnet34'),
                'resnet50': lambda: PSPNet(num_classes=self.num_classes, sizes=(1, 2, 3, 6), psp_size=2048,
                                           deep_features_size=1024, backend='resnet50'),
                'resnet101': lambda: PSPNet(num_classes=self.num_classes, sizes=(1, 2, 3, 6), psp_size=2048,
                                            deep_features_size=1024, backend='resnet101'),
                'resnet152': lambda: PSPNet(num_classes=self.num_classes, sizes=(1, 2, 3, 6), psp_size=2048,
                                            deep_features_size=1024, backend='resnet152')
            }

            self.basemodel = psp_models[backbone]()

        elif self.name == 'deeplab':
            # TODO:(DeepLab) specify output stride
            self.basemodel = DeepLab(backbone=backbone, output_stride=16, num_classes=num_classes)

        else:
            raise ValueError('ERROR: The {} is undefined name of base model.'.format(self.name))

    def forward(self, x):
        outputs = self.basemodel(x)

        return outputs

    def freeze(self, num_freeze_layers):
        # TODO: now works only for PSPNet, UNet and ENet
        if self.name == 'pspnet':
            children = self.basemodel.feats.children()
            
        elif (self.name == 'unet') or (self.name =='enet'):
            children = self.basemodel.children()

        else:
            raise ValueError("ERROR: Can't freeze layers for {} model.".format(self.name))

        for ind, child in enumerate(children):
            if ind < num_freeze_layers:
                for param in child.parameters():
                    param.requires_grad = False



