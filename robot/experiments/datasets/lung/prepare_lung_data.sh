#!/bin/bash
python dirlab_processing.py $1
python prepare_training_data.py --val_dirlab --copd_data_folder $2 --output_folder $3

# sh prepare_lung_data.sh /home/zyshen/data/dirlab_data/copd  /home/zyshen/data/PVT1010  /home/zyshen/experiments/lung_expri