#!/bin/bash

cd ..

python train.py --model_name mot17_scratch --dataset mot17 --dataset_version 17trainval --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1