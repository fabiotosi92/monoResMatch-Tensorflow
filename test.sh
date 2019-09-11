#!bin/bash

image_path=./images/000000_10.png
output_path=./output

#check inference
python main.py --test_single --image_path $image_path --output_path $output_path --save_images
