import os
import argparse

file_dir = os.path.dirname(__file__)


class TrainingOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Object tracking training options")

        # Dataset options
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default='coco',
                                 choices=['coco', 'mot17', 'waymo', 'kitti_tracking'])
        self.parser.add_argument("--data_dir",
                                 type=str,
                                 help="path to the folder where dataset is stored",
                                 default=os.path.join(file_dir, 'data'))
        self.parser.add_argument('--dataset_version', default='')

        # Model options
        self.parser.add_argument("--save_dir",
                                 type=str,
                                 help="directory to save model weights and logs in",
                                 default=os.path.join(file_dir, "models"))
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
        self.parser.add_argument("--model_arch",
                                 type=str,
                                 help="enter the model architecture you'd like to train",
                                 default="dla_34",
                                 choices=['res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                          'dlav0_34 | dla_34 | hourglass'])
        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '64 for resnets and 256 for dla.')
        self.parser.add_argument('--dla_node', default='dcn')
        self.parser.add_argument('--num_head_conv', type=int, default=1)
        self.parser.add_argument('--head_kernel', type=int, default=3, help='')
        self.parser.add_argument('--prior_bias', type=float, default=-4.6)  # -2.19
        self.parser.add_argument('--down_ratio', type=int, default=4, choices=[4])
        self.parser.add_argument('--model_output_list', action='store_true',
                                 help='Used when convert to onnx')

        # Training parameters
        self.parser.add_argument('--num_classes',
                                 type=int,
                                 default=-1,
                                 help="number of object classes to predict. -1 for dataset default"
                                 )
        self.parser.add_argument("--optim",
                                 type=str,
                                 default='adam',
                                 help='training optimzer',
                                 choices=['adam', 'sgd'])
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--lr",
                                 type=float,
                                 help="learning rate",
                                 default=1.25e-4)
        self.parser.add_argument("--num_epochs",
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--lr_step",
                                 type=str,
                                 help="epochs on which to reduce the learning rate by factor of 10",
                                 default='10,15')
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height. -1 for default from dataset",
                                 default=-1)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width. -1 for default from dataset",
                                 default=-1)

        # Logging parameters
        self.parser.add_argument('--no_bar',
                                 action='store_true',
                                 help="Do not display the interactive training bar and simply print losses")
        self.parser.add_argument("--save_period",
                                 type=int,
                                 help="save model every this many epochs'",
                                 default=5)
        self.parser.add_argument('--eval_val', action='store_true')
        self.parser.add_argument('--debug', type=int, default=0,
                                 help='level of visualization.'
                                      '1: only show the final detection results'
                                      '2: show the network output features'
                                      '3: use matplot to display'  # useful when lunching training with ipython notebook
                                      '4: save all visualizations to disk')
        self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                                 help='visualization threshold.')
        self.parser.add_argument('--debugger_theme', default='white',
                                 choices=['white', 'black'])
        self.parser.add_argument('--show_track_color', action='store_true')
        self.parser.add_argument('--not_show_bbox', action='store_true')
        self.parser.add_argument('--not_show_number', action='store_true')
        self.parser.add_argument('--qualitative', action='store_true')
        self.parser.add_argument('--tango_color', action='store_true')


        # Augmentation options
        self.parser.add_argument('--not_rand_crop', action='store_true',
                                 help='not use the random crop data augmentation'
                                      'from CornerNet.')
        self.parser.add_argument('--not_max_crop', action='store_true',
                                 help='used when the training dataset has'
                                      'inbalanced aspect ratios.')
        self.parser.add_argument('--shift', type=float, default=0,
                                 help='when not using random crop, 0.1'
                                      'apply shift augmentation.')
        self.parser.add_argument('--scale', type=float, default=0,
                                 help='when not using random crop, 0.4'
                                      'apply scale augmentation.')
        self.parser.add_argument('--aug_rot', type=float, default=0,
                                 help='probability of applying '
                                      'rotation augmentation.')
        self.parser.add_argument('--rotate', type=float, default=0,
                                 help='when not using random crop'
                                      'apply rotation augmentation.')
        self.parser.add_argument('--flip', type=float, default=0.5,
                                 help='probability of applying flip augmentation.')
        self.parser.add_argument('--no_color_aug', action='store_true',
                                 help='not use the color augmenation '
                                      'from CornerNet')

        # Loss parameters
        self.parser.add_argument('--tracking_weight', type=float, default=1)
        self.parser.add_argument('--reg_loss', default='l1',
                                 help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for bounding box size.')
        self.parser.add_argument('--hp_weight', type=float, default=1,
                                 help='loss weight for human pose offset.')
        self.parser.add_argument('--hm_hp_weight', type=float, default=1,
                                 help='loss weight for human keypoint heatmap.')
        self.parser.add_argument('--amodel_offset_weight', type=float, default=1,
                                 help='Please forgive the typo.')
        self.parser.add_argument('--dep_weight', type=float, default=1,
                                 help='loss weight for depth.')
        self.parser.add_argument('--dim_weight', type=float, default=1,
                                 help='loss weight for 3d bounding box size.')
        self.parser.add_argument('--rot_weight', type=float, default=1,
                                 help='loss weight for orientation.')
        self.parser.add_argument('--nuscenes_att', action='store_true')
        self.parser.add_argument('--nuscenes_att_weight', type=float, default=1)
        self.parser.add_argument('--velocity', action='store_true')
        self.parser.add_argument('--velocity_weight', type=float, default=1)

        # Tracking parameters
        self.parser.add_argument('--ltrb', action='store_true', help='')
        self.parser.add_argument('--ltrb_weight', type=float, default=0.1)
        self.parser.add_argument('--reset_hm', action='store_true')
        self.parser.add_argument('--reuse_hm', action='store_true')
        self.parser.add_argument('--use_kpt_center', action='store_true')
        self.parser.add_argument('--add_05', action='store_true')
        self.parser.add_argument('--dense_reg', type=int, default=1, help='')
        self.parser.add_argument('--pre_hm', action='store_true')
        self.parser.add_argument('--same_aug_pre', action='store_true')
        self.parser.add_argument('--zero_pre_hm', action='store_true')
        self.parser.add_argument('--hm_disturb', type=float, default=0)
        self.parser.add_argument('--lost_disturb', type=float, default=0)
        self.parser.add_argument('--fp_disturb', type=float, default=0)
        self.parser.add_argument('--pre_thresh', type=float, default=-1)
        self.parser.add_argument('--out_thresh', type=float, default=-1)
        self.parser.add_argument('--track_thresh', type=float, default=0.3)
        self.parser.add_argument('--new_thresh', type=float, default=0.3)
        self.parser.add_argument('--max_frame_dist', type=int, default=3)
        self.parser.add_argument('--ltrb_amodal', action='store_true')
        self.parser.add_argument('--ltrb_amodal_weight', type=float, default=0.1)
        self.parser.add_argument('--public_det', action='store_true')
        self.parser.add_argument('--no_pre_img', action='store_true')
        self.parser.add_argument('--zero_tracking', action='store_true')
        self.parser.add_argument('--hungarian', action='store_true')
        self.parser.add_argument('--max_age', type=int, default=-1)
        self.parser.add_argument('--K', type=int, default=100,
                                 help='max number of output objects.')

    def parse(self):
        opt = self.parser.parse_args()

        opt.model_folder = os.path.join(opt.save_dir, opt.model_name)

        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.model_arch else 64

        opt.pad = 127 if 'hourglass' in opt.model_arch else 31
        opt.num_stacks = 2 if opt.model_arch == 'hourglass' else 1

        if opt.debug > 0:
            opt.num_workers = 0
            opt.batch_size = 1

        opt.tracking = True
        opt.pre_img = False if opt.no_pre_img else True
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
