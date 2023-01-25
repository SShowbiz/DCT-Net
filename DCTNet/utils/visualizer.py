"""
Copyright (C) 2019 NVIDIA Corporation. ALL rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os 
import time 
import subprocess
import random
import wandb
import torch

class Visualizer:
    def __init__(self,opt,mode='train'):
        self.opt = opt
        self.name = opt.name 
        self.mode = mode
        self.train_log_dir = os.path.join(opt.checkpoint_path,"logs/%s"%mode)
        self.log_name = os.path.join(opt.checkpoint_path,'loss_log_%s.txt'%mode)
        if opt.local_rank == 0:
            if not os.path.exists(self.train_log_dir):
                os.makedirs(self.train_log_dir)

            wandb.init(project=opt.wandb_project_name, name=opt.task_name)
            wandb.config.update(opt)

            self.log_file = open(self.log_name,"a")
            now = time.strftime("%c")
            self.log_file.write('================ Training Loss (%s) =================\n'%now)
            self.log_file.flush()


    # errors:dictionary of error labels and values
    def plot_current_errors(self,errors,step):
        wandb.log(errors)

    # errors: same format as |errors| of CurrentErrors
    def print_current_errors(self,epoch,i,errors,t):
        message = '(epoch: %d\t iters: %d\t time: %.5f)\t'%(epoch,i,t)
        for k,v in errors.items():

            message += '%s: %.5f\t' %(k,v)

        print(message)

        self.log_file.write('%s\n' % message)
        self.log_file.flush()

    def display_current_results(self, visuals, step, mode, labels):
        if visuals is None:
            return

        assert mode.endswith('TRAIN') or mode.endswith('EVAL') or mode.endswith('TEST')
        if mode.endswith('TRAIN'):
            visual = visuals[random.randrange(0,len(visuals))]
            vis_images = [wandb.Image(images, caption=f"{self.name} {label}") for images, label in zip(visual, labels)]
        elif mode.endswith('EVAL'):
            vis_images = [wandb.Image(images, caption=' / '.join(labels)) for images in visuals]
        else:
            for xs, xg in visuals:
                print(xs.shape)
                print(xg.shape)
                print(torch.cat([xs, xg], dim=0).shape)
            vis_images = [wandb.Image(torch.cat([xs, xg], dim=0), caption=' / '.join(labels)) for xs, xg in visuals]

        wandb.log({mode: vis_images}, step=step)

    def close(self):
        self.log_file.close()   