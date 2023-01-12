import os

import numpy as np
import scipy.ndimage
import PIL.Image
import face_alignment
from PIL import Image, ImageDraw

import argparse
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torchvision import transforms
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default="datasets/custom_anime/")
parser.add_argument('--save_dir', default="datasets/custom_anime_align_jh_landmark/")

def image_align_68(src_file, dst_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=True):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm = np.array(face_landmarks)
        lm_chin          = lm[0  : 17, :2]  # left-right
        lm_eyebrow_left  = lm[17 : 22, :2]  # left-right
        lm_eyebrow_right = lm[22 : 27, :2]  # left-right
        lm_nose          = lm[27 : 31, :2]  # top-down
        lm_nostrils      = lm[31 : 36, :2]  # top-down
        lm_eye_left      = lm[36 : 42, :2]  # left-clockwise
        lm_eye_right     = lm[42 : 48, :2]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60, :2]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68, :2]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        img.save(dst_file, 'PNG')


def image_align_24(src_file, dst_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=True):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm = np.array(face_landmarks)
        lm_chin          = lm[0  : 3, :2]  # left-right
        lm_eyebrow_left  = lm[3  : 6, :2]  # left-right
        lm_eyebrow_right = lm[6  : 9, :2]  # left-right
        lm_nose          = lm[9  : 10, :2]  # top-down
        lm_eye_left      = lm[10 : 15, :2]  # left-clockwise
        lm_eye_right     = lm[15 : 20, :2]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm[20, :2]
        mouth_right  = lm[22, :2]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        print(eye_to_mouth)
        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        # src_file : <PIL>
        img = src_file
        
        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        img.save(dst_file, 'png')


# Cacaded Face Alignment
class CFA(nn.Module):
    def __init__(self, output_channel_num, checkpoint_name=None):
        super(CFA, self).__init__()

        self.output_channel_num = output_channel_num
        self.stage_channel_num = 128
        self.stage_num = 2

        self.features = nn.Sequential(
            nn.Conv2d(  3,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d( 64,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d( 64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),

            # nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))

            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))
        
        self.CFM_features = nn.Sequential(
            #nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, self.stage_channel_num, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        # cascaded regression
        stages = [self.make_stage(self.stage_channel_num)]
        for _ in range(1, self.stage_num):
            stages.append(self.make_stage(self.stage_channel_num + self.output_channel_num))
        self.stages = nn.ModuleList(stages)
        
        # initialize weights
        if checkpoint_name:
            snapshot = torch.load(checkpoint_name)
            self.load_state_dict(snapshot['state_dict'])
        else:
            self.load_weight_from_dict()
    

    def forward(self, x):
        feature = self.features(x)
        feature = self.CFM_features(feature)
        heatmaps = [self.stages[0](feature)]
        for i in range(1, self.stage_num):
            heatmaps.append(self.stages[i](torch.cat([feature, heatmaps[i - 1]], 1)))
        return heatmaps
    

    def make_stage(self, nChannels_in):
        layers = []
        layers.append(nn.Conv2d(nChannels_in, self.stage_channel_num, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(4):
            layers.append(nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(self.stage_channel_num, self.output_channel_num, kernel_size=3, padding=1))
        return nn.Sequential(*layers)


    def load_weight_from_dict(self):
        model_urls = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        weight_state_dict = model_zoo.load_url(model_urls)
        all_parameter = self.state_dict()
        all_weights   = []
        for key, value in all_parameter.items():
            if key in weight_state_dict:
                all_weights.append((key, weight_state_dict[key]))
            else:
                all_weights.append((key, value))
        all_weights = OrderedDict(all_weights)
        self.load_state_dict(all_weights)


if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # path
    # landmark_path = f'align_iamge'
    # os.makedirs(landmark_path, exist_ok=True)

    # param
    num_landmark = 24
    img_width = 256
    checkpoint_name = 'checkpoint_landmark_191116.pth.tar'

    # detector
    face_detector = cv2.CascadeClassifier('lbpcascade_animeface.xml')
    landmark_detector = CFA(output_channel_num=num_landmark + 1, checkpoint_name=checkpoint_name).cuda()

    # transform
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    train_transform = [transforms.ToTensor(), normalize]
    train_transform = transforms.Compose(train_transform)

    images = os.listdir(args.dataset_dir)
    for image_name in images:
        image_path = os.path.join(args.dataset_dir, image_name)
        save_path = os.path.join(args.save_dir, image_name)
        # input image & detect face
        input_img_name = image_path
        img = cv2.imread(input_img_name)
        faces = face_detector.detectMultiScale(img)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(img)
        if faces == ():
            continue

        for x_, y_, w_, h_ in faces:

            # adjust face size
            x = max(x_ - w_ / 8, 0)
            rx = min(x_ + w_ * 9 / 8, img.width)
            y = max(y_ - h_ / 4, 0)
            by = y_ + h_
            w = rx - x
            h = by - y


            # transform image
            img_tmp = img.crop((x, y, x+w, y+h))
            img_tmp = img_tmp.resize((img_width, img_width), Image.BICUBIC)

            draw = ImageDraw.Draw(img_tmp)

            img_tmp_tf = train_transform(img_tmp)
            img_tmp_tf = img_tmp_tf.unsqueeze(0).cuda()

            # estimate heatmap
            heatmaps = landmark_detector(img_tmp_tf)
            heatmaps = heatmaps[-1].cpu().detach().numpy()[0]

            # landmark array
            landmark_array = []

            # calculate landmark position
            for i in range(num_landmark):
                heatmaps_tmp = cv2.resize(heatmaps[i], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
                landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
                landmark_y = landmark[0] 
                landmark_x = landmark[1] 

                # draw landmarks
                draw.ellipse((landmark_x - 2, landmark_y - 2, landmark_x + 2, landmark_y + 2), fill=(255, 0, 0))
                landmark_array.append([landmark_x - 2, landmark_y - 2, landmark_x + 2, landmark_y + 2])

        # output image
        #img_tmp.save(f'./LineWebtoonCharacterDataset/MyDeepestSecret/Emma/align2-{str(k).zfill(4)}.png')

        image_align_24(img_tmp, save_path, landmark_array, transform_size=256)