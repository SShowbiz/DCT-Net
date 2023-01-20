import sys
import os
import argparse

import numpy as np
import scipy.ndimage
import PIL.Image
import face_alignment
from warp import compute_h_norm
import torch
from PIL import Image, ImageDraw
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default="datasets/arcane/")
parser.add_argument('--save_dir', default="datasets/arcane_256/")

def image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=1024, enable_padding=True):
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

        draw = ImageDraw.Draw(img)

        # for landmark in lm[36 : 48, :2]:
        #     landmark_x, landmark_y = landmark
        #     draw.ellipse((landmark_x - 2, landmark_y - 2, landmark_x + 2, landmark_y + 2), fill=(255, 0, 0))
        
        # for landmark in lm[27 : 36, :2]:
        #     landmark_x, landmark_y = landmark
        #     draw.ellipse((landmark_x - 2, landmark_y - 2, landmark_x + 2, landmark_y + 2), fill=(0, 0, 255))

        # for landmark in lm[48 : 68, :2]:
        #     landmark_x, landmark_y = landmark
        #     draw.ellipse((landmark_x - 2, landmark_y - 2, landmark_x + 2, landmark_y + 2), fill=(0, 255, 0))

        # for landmark in lm[17 : 27, :2]:
        #     landmark_x, landmark_y = landmark
        #     draw.ellipse((landmark_x - 2, landmark_y - 2, landmark_x + 2, landmark_y + 2), fill=(0, 0, 0))
        
        # for landmark in lm_chin:
        #     landmark_x, landmark_y = landmark
        #     draw.ellipse((landmark_x - 2, landmark_y - 2, landmark_x + 2, landmark_y + 2), fill=(127, 127, 127))
        
        main_landmarks = np.vstack([eye_left, eye_right, mouth_avg])
        dst_file_name = dst_file.split('.')[0]
        # img.save(f'{dst_file_name}_original.jpg', 'PNG')

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink
            main_landmarks /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]
            main_landmarks -= crop[0:2]

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
            main_landmarks += pad[0:2]

        quad = quad + 0.5
        # Transform.

        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, quad.flatten(), PIL.Image.BILINEAR)
        H = compute_h_norm(np.array([quad[0], quad[3], quad[2], quad[1]]), np.array([[0, 0], [transform_size, 0], [transform_size, transform_size], [0, transform_size]]))
        H_inverse = np.linalg.inv(H)

        warped_homogenous_landmarks = H_inverse @ np.concatenate((main_landmarks, np.ones((3, 1))), axis=1).T
        warped_landmarks = (warped_homogenous_landmarks / warped_homogenous_landmarks[2])[:2].T

        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Re-Scaling
        left_eye, right_eye, mouth = warped_landmarks
        eye_to_eye_length = np.linalg.norm(right_eye - left_eye)
        avg_eye_to_mouth_length = np.linalg.norm(mouth - (left_eye + right_eye)*0.5)

        # img.save(dst_file, 'PNG')

        if eye_to_eye_length > avg_eye_to_mouth_length:
            w, h = img.size
            center_height = (right_eye[1] + left_eye[1])*0.5
            ratio = eye_to_eye_length / avg_eye_to_mouth_length

            img = img.resize((output_size, int(ratio * output_size)), PIL.Image.BICUBIC)
            img = img.crop((0, int(center_height * ratio - output_size * 0.5), w, int(center_height * ratio + output_size * 0.5)))
            img.save(f'{dst_file_name}_resize.jpg', 'PNG')


        else:
            w, h = img.size
            center_width = (right_eye[0] + left_eye[0])*0.5

            ratio = avg_eye_to_mouth_length / eye_to_eye_length
            img = img.resize((int(ratio * output_size), output_size), PIL.Image.BICUBIC)
            img = img.crop((int(center_width * ratio - output_size * 0.5), 0, int(center_width * ratio + output_size * 0.5), h))

            img.save(f'{dst_file_name}_resize.jpg', 'PNG')


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

    available_images = os.listdir('datasets/align_test_cp/')
    # images_candidates = os.listdir('datasets/danbooru2019_cp/')

    # if not os.path.exists('datasets/danbooru2019_cp_align/'):
            # os.makedirs('datasets/danbooru2019_cp_align/')
    available_dict = {f"{image.split('_')[0]}.jpg": True for image in available_images}
    

    images = os.listdir(args.dataset_dir)
    for image_name in tqdm(images):
        if available_dict.get(image_name):
            image_path = os.path.join(args.dataset_dir, image_name)
            landmarks = landmarks_detector.get_landmarks(image_path)
            
            if landmarks is None:
                pass
            else:
                for i, face_landmarks in enumerate(landmarks, start=1):
                    aligned_face_path = os.path.join(args.save_dir, image_name)
                    image_align(image_path, aligned_face_path, face_landmarks)