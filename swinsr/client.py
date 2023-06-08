import base64
import requests
import json
import os
import cv2
import os.path as osp
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO

url = 'http://127.0.0.1:9999/runImageSR'
    
def base64toPIL(respond):
    img_base64 = base64.b64decode(respond)  # b64 str
    img = Image.open(BytesIO(img_base64)).convert('RGB')
    return img
def Test():
    imgdir = './test/poster/large_poster/福邸.png'
    save_compress_sr_dir = './test/poster/large_poster_compress_sr'

    if not osp.exists(save_compress_sr_dir):
        os.makedirs(save_compress_sr_dir)

    with open(imgdir, 'rb') as f:
        img_byte = base64.b64encode(f.read())
        img_str = img_byte.decode('ascii') #对byte进行str解码

    service_data = {'image': img_str, 'poster_type':'large'}
    respond = requests.post(url, data=json.dumps(service_data)).json()
    try:
        respond_code = respond['respond_code']
        respond_result = respond['respond_result']
        respond_msg = respond['respond_message']
        # decode, save
        predict_compress_img = base64toPIL(respond_result['compress_sr_out'])
        predict_compress_img.save(osp.join(save_compress_sr_dir, imgdir.split('/')[-1].split('.')[0]+'.jpg'))
        
        if respond_code == 200:
            print('Running sr {}'.format(respond_msg))
        else:
            print('respond_code:{}, respond_message:{}'.format(respond_code, respond_msg))
    except:
        respond_code = respond['respond_code']
        respond_msg = respond['respond_message']
        print('respond_code:{}  respond_message:{}'.format(respond_code, respond_msg))
if __name__ == '__main__':
    Test()
    
    