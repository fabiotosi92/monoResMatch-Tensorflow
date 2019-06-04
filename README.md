# monoResMatch-Tensorflow
**Learning monocular depth estimation infusing traditional stereo knowledge**

[Fabio Tosi](https://vision.disi.unibo.it/~ftosi/), [Filippo Aleotti](https://vision.disi.unibo.it/~faleotti/), [Matteo Poggi](https://vision.disi.unibo.it/~mpoggi/) and [Stefano Mattoccia](https://vision.disi.unibo.it/~smatt/Site/Home.html)   
CVPR 2019


[Paper](https://vision.disi.unibo.it/~ftosi/papers/monoResMatch.pdf)   
[Supplementary material](https://vision.disi.unibo.it/~ftosi/papers/monoResMatch_supp.pdf)   
[Poster](https://vision.disi.unibo.it/~ftosi/papers/monoResMatch_poster.pdf)  
[Youtube Video](https://www.youtube.com/watch?v=h6Wo5MqbCY0&t=2s)

## Architecture
Tensorflow implementation of monocular Residual Matching (monoResMatch) network.

![Alt text](./images/architecture.png?raw=true "architecture")


## Requirements
This code was tested with Tensorflow 1.8, CUDA 9.0 and Ubuntu 16.04. 


## Training

Cityscapes

The CityScapes dataset contains stereo pairs concerning about 50 cities in Germany taken from amoving vehicle in various weather conditions. It consists of 22,973 stereo pairs splitted into train, validation and test sets. You can find the training set file in ./utils/filenames/cityscapes_train_files.txt 

You will need to register in order to download the data. 

```shell
python main.py  --is_training \
                --data_path_image [path_cityscapes] \
                --data_path_proxy [path_cityscapes_proxy] \
                --filenames_file ./utils/filenames/cityscapes_train_files.txt \
                --batch_size 6 \
                --iterations 150000 \
                --learning_rate_schedule 100000,120000 \
                --patch_width 512 \
                --patch_height 256 \ 
                --height 512 \ 
                --width 1024 \ 
                --initial_learning_rate 0.0001 \ 
                --log_directory ./log/CS \ 
                --dataset cityscapes
```


KITTI

We used the Eigen split of the data amounting 22600 training samples. You can find it in ./utils/filenames/eigen_train_files.txt folder.
You can download the entire full kitti dataset by running:
```shell
wget -i utils/kitti_archives_to_download.txt -P [kitti_path]
```

```shell
python main.py  --is_training \
                --data_path_image [path_kitti] \ 
                --data_path_proxy [path_kitti_proxy] \
                --filenames_file [kitti_train_file] \
                --batch_size 6 \ 
                --iterations 300000 \
                --learning_rate_schedule 180000,240000 \
                --patch_height 192 \ 
                --patch_width 640 \ 
                --height 384 \ 
                --width 1280 \ 
                --initial_learning_rate 0.0001 \ 
                --log_directory ./log/K 
```

You can also load an existing model using ``` --checkpoint_path ``` or/and fine-tune the network using the ``` --retrain ``` flag.

```shell
python main.py  --is_training \
                --data_path_image [path_kitti] \ 
                --data_path_proxy [path_kitti_proxy] \
                --filenames_file [kitti_train_file] \
                --batch_size 6 \ 
                --iterations 300000 \
                --learning_rate_schedule 180000,240000 \
                --patch_height 192 \ 
                --patch_width 640 \ 
                --height 384 \ 
                --width 1280 \ 
                --initial_learning_rate 0.0001 \ 
                --log_directory ./log/CS_K \ 
                --checkpoint_path ./log/CS/model-150000 \
                --retrain
```

**Warning:** If you want to fine-tune on KITTI raw LiDAR measurements you need to convert depth values to disparities using the baseline distance between the cameras and the camera focal length. 

## Testing

KITTI Eigen test split
```shell
python main.py --output_path [output_path] \ 
               --data_path_image [path_kitti] \
               --filenames_file ./utils/filenames/eigen_test_files.txt \
               --checkpoint_path ./log/CS_K/model-300000
```
You can also save output images in png format enabling the --save_images flag. 

If you want to try on a single image:

```shell
python main.py --test_single \
               --image_path [image_path] \
               --output_path [output_path] \ 
               --checkpoint_path ./log/CS_K/model-300000
```
## Evaluation

To evaluate run:

```shell
python utils/evaluate_kitti.py --split eigen \
                               --disp_folder [path_test_npy] \
                               --gt_path [path_kitti] \ 
                               --garg_crop
```

## Proxy label generation

You can use the code available at https://github.com/ivankreso/stereo-vision/tree/master/reconstruction/base/rSGM to generate SGM proxy labels and train the monoResMatch network. 

**Warning:** Proxy labels and image stereo pairs should be saved in a folder having the same structure specified in the training file. Pay attention that image stereo pairs are .jpg files whilst proxy labels are 16 bit .png files. 



## Pretrained models

You can download the following pre-trained models:


* [KITTI](https://drive.google.com/open?id=1Uw8nKX-0J3D0y8XN9wMN3lzgx9rGpdTy)

* [Cityscapes -> KITTI](https://drive.google.com/open?id=1kM_HGcIug_a4CczYHidXvpoUFuqnLozN)

* [Cityscapes -> KITTI 100-raw](https://drive.google.com/open?id=1hMf7e3Nl709SMzH1DoKAfOt4_dAgcrhw)

* [Cityscapes -> KITTI 200-acrt](https://drive.google.com/open?id=1SxZH4TJpx7WCdearGHuWfui_DCmRbK6n)

* [Cityscapes -> KITTI 200-acrt + 100-raw](https://drive.google.com/open?id=1PDkpgvQjKL4DefU02LcZlUmDlMZRlW0i)

* [Cityscapes -> KITTI 200-acrt + 200-raw](https://drive.google.com/open?id=10idprVLQmrcEJWNq9ea95UvPhow1Azb2)

* [Cityscapes -> KITTI 200-acrt + 500-raw](https://drive.google.com/open?id=1C_Due2smrC1S8qZV6azLCoQY2tWO_aW0)

* [Cityscapes -> KITTI 200-acrt + 700-raw](https://drive.google.com/open?id=1jNVCOvbhjW6HhH5h7fK2p3RKrDH868D7)


## Some qualitative results
Qualitative results of the proposed depth-from-mono architecture. From left to right, the input image from KITTI 2015 test set (a), the predicted depth by monoResMatch fine-tuned using (b) 200-acrt ground-truth labels, (c) 200-acrt + 700 raw LiDAR samples and (d) SGM only.


![Alt text](./images/qualitative.png?raw=true "supplementary")


## Citation
If you find this code useful in your research, please cite:

```shell
@InProceedings{Tosi_2019_CVPR,
author = {Tosi, Fabio and Aleotti, Filippo and Poggi, Matteo and Mattoccia, Stefano},
title = {Learning monocular depth estimation infusing traditional stereo knowledge},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Acknowledgements

The evaluation code is from "Unsupervised Monocular Depth Estimation with Left-Right Consistency, by C. Godard, O Mac Aodha, G. Brostow, CVPR 2017".
