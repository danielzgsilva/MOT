import os
import argparse

file_dir = os.path.dirname(__file__)


class TrainingOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Object tracking training options")

        # Dataset
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default='coco',
                                 choices=['coco', 'mot17', 'waymo'])
        self.parser.add_argument("--data_dir",
                                 type=str,
                                 help="path to the folder where dataset is stored",
                                 default=os.path.join(file_dir, 'data'))
        self.parser.add_argument('--dataset_version', default='')

        # Model
        self.parser.add_argument("--save_dir",
                                 type=str,
                                 help="directory to save model weights and logs in",
                                 default=os.path.join(file_dir, "models"))
        self.parser.add_argument("--model_arch",
                                 type=str,
                                 help="enter the model architecture you'd like to train",
                                 default="dla34",
                                 choices=['res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                          'dlav0_34 | dla_34 | hourglass'])
        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '64 for resnets and 256 for dla.')
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="name to save this model as")
        self.parser.add_argument("--load_model",
                                 type=str,
                                 default='',
                                 help="pretrained model to load")
        self.parser.add_argument("--resume",
                                 action='store_true',
                                 help="resume training from last save")

        # Input size
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height. -1 for default from dataset",
                                 default=-1)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width. -1 for default from dataset",
                                 default=-1)

        # Training parameters
        self.parser.add_argument('--num_classes',
                                 type=int,
                                 default=-1)
        self.parser.add_argument("--optim",
                                 type=str,
                                 default='adam',
                                 help='training optimzer',
                                 options=['adam', 'sgd'])
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=32)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1.25e-4)
        self.parser.add_argument("--num_epochs",
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--lr_step",
                                 type=str,
                                 help="epochs on which to reduce the learning rate by factor of 10",
                                 default='10')
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)
        self.parser.add_argument("--save_period",
                                 type=int,
                                 help="save model every this many epochs'",
                                 default=5)

        self.parser.add_argument('--reset_hm', action='store_true')
        self.parser.add_argument('--reuse_hm', action='store_true')

        # Loss
        self.parser.add_argument('--tracking_weight',
                                 type=float,
                                 default=1)
        self.parser.add_argument('--reg_loss',
                                 default='l1',
                                 help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight',
                                 type=float,
                                 default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight',
                                 type=float,
                                 default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight',
                                 type=float,
                                 default=0.1,
                                 help='loss weight for bounding box size.')

        # Tracking
        self.parser.add_argument('--pre_thresh', type=float, default=-1)
        self.parser.add_argument('--track_thresh', type=float, default=0.3)
        self.parser.add_argument('--new_thresh', type=float, default=0.3)

    def parse(self):
        opt = self.parser.parse_args()

        opt.model_folder = os.path.join(opt.save_dir, opt.model_name)

        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 64

        opt.pad = 127 if 'hourglass' in opt.arch else 31
        opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

        opt.tracking = True
        opt.out_thresh = max(opt.track_thresh, opt.out_thresh)
        opt.pre_thresh = max(opt.track_thresh, opt.pre_thresh)
        opt.new_thresh = max(opt.track_thresh, opt.new_thresh)

        return opt

class TestingOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Object tracking testing options")

        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default='coco',
                                 choices=['coco', 'mot17', 'waymo'])
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, 'cityscapes'))

        self.parser.add_argument("--model_path",
                                 type=str,
                                 help="directory where models are saved in",
                                 default=os.path.join(file_dir, "models"))

        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="name of the model file")

        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height. -1 for default from dataset",
                                 default=-1)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width. -1 for default from dataset",
                                 default=-1)

        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=32)

        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
