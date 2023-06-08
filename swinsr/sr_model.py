import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
from omegaconf import OmegaConf
import yaml
from .models.network_sr import SR_Real_Gan as net
from .utils import util_calculate_psnr_ssim as util

import os.path as osp
from tqdm import tqdm
import time
import random
from PIL import Image
import base64
from io import BytesIO

seed = 42
torch.manual_seed(seed)

class Img_SR_Model(object):
    def __init__(self, config_dir,scale):
        self.config = self.load_config(config_dir)
        self.device = self.config.device 
        self.model_path = self.config.model_path
        self.scale = scale
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.define_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.window_size = 8
        print('build model sucessful')

    def load_config(self, config_dir):
        with open(config_dir, 'r', encoding='UTF-8') as f:
            config = OmegaConf.create(yaml.safe_load(f))
        return config

    def define_model(self):
        model = net(upscale=self.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')

        # model = net(upscale=self.scale, in_chans=3, img_size=64, window_size=8,
        #                 img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
        #                 num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        #                 mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')

        param_key_g = 'params_ema'
        pretrained_model = torch.load(self.model_path)
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        return model

    def compress(self, img, h_lr, w_lr, type='small'):
        # h_sr, w_sr = img.shape[:2]
        if type == 'small':
            h_out, w_out = 260, 700
            img = cv2.resize(img, (w_out, h_out))
        elif type == 'large':
            h_out, w_out = h_lr, w_lr
            img = cv2.resize(img, (w_out, h_out))
        else:
            h_out, w_out = img.shape[0], img.shape[1]

        new_img_path = osp.join('test/saveimg', str(random.randint(0, 10000)) + '.jpg')
        quality = 95
        while quality > 0:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            cv2.imencode('.jpg', img, params)[1].tofile(new_img_path)
            file_size = os.stat(new_img_path).st_size / 1000  # kb
            quality -= 1
            if file_size < 150:
                break
        img = cv2.imdecode(np.fromfile(new_img_path, dtype=np.uint8), -1)
        return img, new_img_path
    

    def compress_PIL(self, img, h_lr, w_lr, type='small'):
        if type == 'small':
            h_out, w_out = 260, 700
            img = cv2.resize(img, (w_out, h_out))
        elif type == 'large':
            h_out, w_out = h_lr, w_lr
            img = cv2.resize(img, (w_out, h_out))
        else:
            h_out, w_out = img.shape[0], img.shape[1]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR TO RGB
        img = Image.fromarray(img) # To PIL

        random_name = str(random.randint(0, 10000)) + '.jpg'
        save_compress_path = osp.join('test/savecompress', random_name)
        save_inter_path = osp.join('test/saveinter', random_name)
        
        quality = 95
        while True:
            img.save(save_compress_path, quality=quality)
            inter_img = Image.open(save_compress_path)
            inter_img.save(save_inter_path)
            inter_size = os.path.getsize(save_inter_path) / 1024 #os.stat(new_img_path).st_size / 1000
            if inter_size < 150:
                break
            else:
                quality -= 2

        with open(save_inter_path, 'rb') as file:  # 转换图片成base64格式
            data = file.read()
            encodestr = base64.b64encode(data)
            compress_img = str(encodestr, 'utf-8')

        return compress_img, save_compress_path, save_inter_path

    
    def inference(self, img_lq):
        h_lr, w_lr = img_lq.shape[:2]
        img_lq = img_lq.astype(np.float32) / 255.
        img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1)) # HWC-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = self.model(img_lq)
            output = output[..., :h_old * self.scale, :w_old * self.scale]
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        print(f'Swin_shape:{output.shape}')
        # compress image quality to 150kb
        # compress_output, compress_path, inter_path = self.compress_PIL(output, h_lr, w_lr, type=img_type)
        # os.remove(compress_path)
        # os.remove(inter_path)
        # output = {'compress_sr_out':compress_output}

        return output

if __name__=='__main__':

    # imgdir = 'test/poster/large_poster'
    imgdir = '/content/CodeFormer/inputs/input_test'
    config_path = 'config.yaml'
    savedir_poster = 'test/poster/large_poster_sr'
    # savedir_compress = 'test/poster/large_poster_compress'

    if not osp.exists(savedir_poster):
        os.makedirs(savedir_poster)
    # if not osp.exists(savedir_compress):
    #     os.makedirs(savedir_compress)

    model = Img_SR_Model(config_path)

    for f in tqdm(os.listdir(imgdir)):
        # img = cv2.imread(osp.join(imgdir, f))
        img = cv2.imdecode(np.fromfile(osp.join(imgdir, f), dtype=np.uint8), -1)
        sr = model.inference(img)
        cv2.imencode('.jpg', sr)[1].tofile(osp.join(savedir_poster, f))
        

 

    








