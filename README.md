# DCT-Net

## Environment Setting

```shell
$ pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install wandb
$ pip install tqdm
$ pip install opencv-python
```

<br/>

## Weight & Bias Setting

This repository uses wandb for log images. To use the feature, sign in wanb and use below command for initialize your account.

```shell
$ wandb login ${YOU_API_KEY}
```

<br/>

## Training Network (CCN)

### Download Pretrained Models

1. StyleGAN checkpoints

    To download pretrained StyleGAN2 checkpoint, follow the instruction written in [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch#convert-weight-from-official-checkpoints). Place  checkpoint file into `pretrain_models/stylegan2-ffhq-config-f.pt`

2. ID Loss Module

    To download ID Loss module, use the command below. 

    ```shell
    $ wget https://github.com/LeslieZhoa/DCT-NET.Pytorch/releases/download/v0.0/model_ir_se50.pth -P ../pretrain_models
    ```

    The above command will download the checkpoint file and place into `pretain_models/` path.

<br/>

### Preparing Data

For training CCN, FFHQ-Aligned few shot (50 ~ 100) facial images with $1024 \times 1024$ size are needed. 

It is recommended to use additional implementations such as [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and [SwinIR](https://github.com/JingyunLiang/SwinIR) rather than using naive interpolation-based resizing algorithm. 

However, the command below can be helpful to resize dataset to downsample using large scale (e.g. $\times 4$, $\times 8$) super resolution model 

```shell
$ python resize_datasets.py --dataset_dir ${DIR_YOUR_ORIGINAL_DATASET} --save_dir ${DIR_YOUR_RESIZED_DATASET} --size 1024
```

Place the images in `DCTNet/datasets/CCN/${DIR_YOUR_DATASET}`

<br/>

### Training

For training CCN, use the below command. The appropriate number of itations varies depending on the dataset, but it is recommended to use about 1000.

```shell
$ python DCTNet/train.py \
--model ccn --batch_size 8 \
--checkpoint DCTNet/checkpoints/CCN/${DIR_FOR_SAVE_CKPT}/ \
--root DCTNet/datasets/CCN/${DIR_YOUR_DATASET} \
--lr 0.002 --print_interval 50 --save_interval 50 \
--task_name anime_ccn_titan \
```

Feel free to terminate the command at discretion.

<br/>

### Blend Models

The trained checkpoint will be saved in the format of `DCTNet/checkpoints/CCN/${DIR_FOR_SAVE_CKPT}/StyleGAN/00X-000XXXXX.pth`. Model Blending may be used to secure a variety of data to be generated by CCN. 

```
$ python blend_models.py --model_path checkpoints/CCN/${DIR_FOR_SAVE_CKPT}/StyleGAN/00X-000XXXXX.pth --level 2 --blend_width 0
```

The above command generates blended model in same directory path `DCTNet/checkpoints/CCN/${DIR_FOR_SAVE_CKPT}/StyleGAN/`. Generally low level, high blend_width blended models generate images similar to targe domain images.

<br/>

### (Optional) Change Strength of Gaussian Truncation

The StyleGAN model often generates artifacts in some input noises. Gaussian Truncation is applied in default, but the strength can be change with config variable `truncation` in `model/styleganModule/config.py`.

Low truncation value can cut off the noises which model is not experienced (has high probability to generate artifacts), but it also hurts the variation of generated images.

<br/>

### Generate TTN Training Images using Blended Model

TTN dataset images can be generated using blended model. It is needed to generate images for training (~10,000) and validation (~3,000) To train TTN. The number of image generating can be changed by modifying `mx_gen_iters` config variable in `model/styleganModule/config.py`.

```shell
$ cd DCTNet/utils/
$ python get_tcc_input.py --model_path ../checkpoints/CCN/${DIR_FOR_SAVE_CKPT}/StyleGAN/${BLENDED_MODEL_CKPT} --output_path ../datasets/TTN/${DIR_FOR_SAVE_TTN_DATASETS}
```