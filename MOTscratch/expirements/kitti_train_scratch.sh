#!/bin/bash

python train.py --model_name kitti_scratch --dla_node conv --dataset kitti_tracking --dataset_version train --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1
