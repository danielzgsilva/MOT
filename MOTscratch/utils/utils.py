import torch

# from datasets.coco import COCO
# from .datasets.kitti import KITTI
# from .datasets.coco_hp import COCOHP
from datasets.mot import MOT
from datasets.kitti_tracking import KITTITracking
# from .datasets.nuscenes import nuScenes
# from .datasets.crowdhuman import CrowdHuman
# from .datasets.kitti_tracking import KITTITracking
# from .datasets.custom_dataset import

from networks.dla import DLASeg
from losses import GenericLoss


# from .networks.resdcn import PoseResDCN
# from .networks.resnet import PoseResNet
# from .networks.dlav0 import DLASegv0
# from .networks.generic_network import GenericNetwork

def get_optimizer(optim, lr, model):
    if optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=0.0001)
    else:
        assert 0, optim

    return optimizer


def get_dataset(dataset):
    dataset_dict = {
        # 'custom': CustomDataset,
        # 'coco': COCO,
        # 'kitti': KITTI,
        # 'coco_hp': COCOHP,
        'mot': MOT,
        # 'nuscenes': nuScenes,
        # 'crowdhuman': CrowdHuman,
        'kitti_tracking': KITTITracking
    }

    return dataset_dict[dataset]


def get_model(arch, opt):
    network_dict = {
        'dla': DLASeg
        # 'resdcn': PoseResDCN,
        # 'res': PoseResNet,
        # 'dlav0': DLASegv0,
        # 'generic': GenericNetwork
    }

    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch

    try:
        model_class = network_dict[arch]
        model = model_class(num_layers, heads=opt.heads, head_convs=opt.head_conv, opt=opt)
        return model

    except KeyError:
        print('chosen architecture {} is not supported'.format(arch))


def get_losses(opt):
    loss_order = ['hm', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp',
                  'hp_offset', 'dep', 'dim', 'rot', 'amodel_offset',
                  'ltrb_amodal', 'tracking', 'nuscenes_att', 'velocity']

    loss_states = ['tot'] + [k for k in loss_order if k in opt.heads]
    loss = GenericLoss(opt)

    return loss_states, loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def update_dataset_and_head_info(opt, dataset):
    opt.num_classes = dataset.num_categories \
        if opt.num_classes < 0 else opt.num_classes

    input_height, input_width = dataset.default_resolution
    input_height = opt.height if opt.height > 0 else input_height
    input_width = opt.width if opt.width > 0 else input_width

    opt.input_size = (input_height, input_width)
    opt.height, opt.width = input_height, input_width
    opt.output_h, opt.output_w = input_height // opt.down_ratio, input_width // opt.down_ratio

    opt.heads = {'hm': opt.num_classes,
                 'reg': 2,
                 'wh': 2,
                 'tracking': 2}

    weight_dict = {'hm': opt.hm_weight,
                   'wh': opt.wh_weight,
                   'reg': opt.off_weight,
                   'tracking': opt.tracking_weight,
                   'ltrb': opt.ltrb_weight,
                   'ltrb_amodal': opt.ltrb_amodal_weight}

    if opt.ltrb:
        opt.heads.update({'ltrb': 4})
    if opt.ltrb_amodal:
        opt.heads.update({'ltrb_amodal': 4})

    opt.weights = {head: weight_dict[head] for head in opt.heads}
    for head in opt.weights:
        if opt.weights[head] == 0:
            del opt.heads[head]

    opt.head_conv = {head: [opt.head_conv
                            for i in range(opt.num_head_conv if head != 'reg' else 1)]
                     for head in opt.heads}
    return opt


def load_model(model, model_path, opt, optimizer=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if (state_dict[k].shape != model_state_dict[k].shape) or \
                    (opt.reset_hm and k.startswith('hm') and (state_dict[k].shape[0] in [80, 1])):
                if opt.reuse_hm:
                    print('Reusing parameter {}, required shape{}, ' \
                          'loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    if state_dict[k].shape[0] < state_dict[k].shape[0]:
                        model_state_dict[k][:state_dict[k].shape[0]] = state_dict[k]
                    else:
                        model_state_dict[k] = state_dict[k][:model_state_dict[k].shape[0]]
                    state_dict[k] = model_state_dict[k]
                else:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and opt.resume:
        if 'optimizer' in checkpoint:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = opt.lr
            for step in opt.lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
