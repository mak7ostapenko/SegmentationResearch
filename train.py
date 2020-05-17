import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.utils.utils import get_config
from src.nn.metrics.multi_loss import MultiLoss
from src.nn.data.generator.rope import RopeRobot
from src.nn.models.model import SegmentationModel
from src.nn.models.trainer import SegmentationTrainer
import src.nn.data.transforms.segmentation as aug
from src.nn.data.generator.pascal_voc import PascalVOC


def get_data_pascal_loaders(batch_size, image_height, image_width):
    local_path = '/researh/dataset/VOCdevkit/VOC2012'

    train_transforms = aug.Compose([aug.RandomCrop(500),
                                    aug.AdjustGamma(0.2),
                                    aug.AdjustSaturation(0.5),
                                    aug.AdjustHue(0.5),
                                    aug.AdjustBrightness(0.5),
                                    aug.AdjustContrast(0.9),
                                    aug.RandomRotate(10),
                                    aug.RandomHorizontallyFlip()
                                    ])

    train_dataset = PascalVOC(root=local_path,
                              split='train',
                              is_transform=True,
                              img_size=(image_height, image_width),
                              augmentations=train_transforms)
    test_dataset = PascalVOC(root=local_path,
                             split='val',
                             is_transform=True,
                             img_size=(image_height, image_width))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader


def get_data_loaders(model_type, num_classes, batch_size, image_height, image_width, roi=None):
    #train_frame_dir = 'dataset/blade_segm/train.csv'
    #test_frame_dir = 'dataset/blade_segm/test.csv'

    train_frame_dir = 'dataset/hand_parts_segm/train_spot.csv'
    test_frame_dir = 'dataset/hand_parts_segm/test_spot.csv'
    # TOiDO
    #train_frame_dir = './dataset/hand_fixer_segm/train_spot.csv'
    #test_frame_dir = './dataset/hand_fixer_segm/test_spot.csv'

    train_transform = aug.Compose([
        aug.RandomCrop(700),
       # aug.AdjustGamma(0.8), 
        aug.AdjustSaturation(0.8),
        aug.AdjustHue(0.3),
        aug.AdjustBrightness(0.5),
        aug.AdjustContrast(0.9),
        aug.ResizePIL((image_height, image_width)),
        aug.RandomRotate(45),
        aug.RandomHorizontallyFlip(),

        aug.ToTensor(),
        aug.Normalize([0, 0, 0], [1, 1, 1]),
    ])

    test_transform = aug.Compose([aug.ResizePIL((image_height, image_width)), 
        aug.ToTensor(),
        aug.Normalize([0, 0, 0], [1, 1, 1])])

    if roi is None:
        adaptive_roi = False
    else:
        adaptive_roi = True

    # NOTE: target_type="gray_scale" for dispenser target segmentation, depends on data
    # NOTE: target_type="binary_RGB" for blade segmentation
    train_dataset = RopeRobot(train_frame_dir, num_classes=num_classes, transform=train_transform, target_type='gray_scale',
                              default_roi=roi, adaptive_roi=adaptive_roi, model_type=model_type)
    test_dataset = RopeRobot(test_frame_dir, num_classes=num_classes, transform=test_transform, target_type='gray_scale',
                             default_roi=roi, adaptive_roi=adaptive_roi, model_type=model_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader


def main():
    # GET CONFIGS
    # TODO: for each model specify training path
    # config_dir='configs/rope/inference_pipeline/pspnet_first_stage.json'
    #config_dir='configs/rope/inference_pipeline_1/enet_hand_fixer_segm.json'
    #config_dir = 'configs/rope/inference_pipeline/pspnet_hand_fixer_segm_.json'
    #config_dir = 'configs/rope/inference_pipeline_1/enet_blade_segm.json'
    config_dir = 'configs/rope/inference_pipeline_1/enet_hand_parts_segm.json' 
    config = get_config(config_dir)

    torch.manual_seed(config['seed'])

    lr = config['lr']
    batch_size = config['batch_size']
    num_epoch = config['num_epoch']

    loss_names = config['loss_names']
    loss_weights = config['loss_weights']

    log_dir = config['log_dir']
    log_interval = config['log_interval']

    scale = config['scale']
    num_classes = config['num_classes']
    image_height = config['image_height']
    image_width = config['image_width']
    roi = config['roi']

    if roi is not None:
        roi = tuple(roi)
    print('train.py: roi = ', roi)

    model_name = config['model_name']
    model_type = config['model_type']
    checkpoint_path = config['checkpoint_path']
    pretrained = config['pretrained']
    backbone = config['backbone']
    num_input_channels = config['num_input_channels']

    image_shape = (image_height, image_width, num_input_channels)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device is ', device)

    # PREPARE DATA
    train_loader, test_loader = get_data_loaders(model_type, num_classes, batch_size, image_height, image_width, roi)

    # PREPARE MODEL
    model = SegmentationModel(num_input_channels=num_input_channels,
                              num_classes=num_classes,
                              name=model_name,
                              backbone=backbone,
                              pretrained=pretrained)
    print('model_name = ', model_name)
    # TODO: freeze layers
    model.freeze(num_freeze_layers=13)
    model = model.to(device)

    # PREPARE EVALUATION AND OPTIMIZATION PARTS FOR TRAINING
    optimizer = Adam(params=filter(lambda param: param.requires_grad, model.basemodel.parameters()))
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-8)

    multi_loss = MultiLoss(loss_names, loss_weights, device, num_classes)

    # INITIALIZE TRAINING
    trainer = SegmentationTrainer(model, train_loader, test_loader,
                                  optimizer, multi_loss, device,
                                  log_interval, num_epoch, log_dir,
                                  checkpoint_path, pretrained, image_shape,
                                  scale)
    # TRAIN
    trainer.train()


if __name__ == '__main__':
    main()


