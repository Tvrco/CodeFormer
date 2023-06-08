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

def decodeRespond(respond):
    """
    respond:str
    """
    respond = respond.encode('ascii')  # byte
    img_base64 = base64.b64decode(respond)  # b64 str
    img = np.asarray(bytearray(img_base64), dtype='uint8')  # array
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED) # cv2
    return img
    
def base64toPIL(respond):
    img_base64 = base64.b64decode(respond)  # b64 str
    img = Image.open(BytesIO(img_base64)).convert('RGB')
    return img

if __name__ == '__main__':
    # Test()
    
    imgdir = './test/poster/large_poster'
    # save_sr_dir = './test/poster/small_poster_sr_x4'
    save_compress_dir = './test/poster/large_poster_compress'

    # if not osp.exists(save_sr_dir):
    #     os.makedirs(save_sr_dir)
    if not osp.exists(save_compress_dir):
        os.makedirs(save_compress_dir)

    for file in tqdm(sorted(os.listdir(imgdir))):
        if file != '.ipynb_checkpoints':
            print('process:', file)
            imgpath = osp.join(imgdir, file)
            with open(imgpath, 'rb') as f:
                img_byte = base64.b64encode(f.read())
                img_str = img_byte.decode('ascii') #对byte进行str解码
            service_data = {'image': img_str, 'poster_type':'large'}
            respond = requests.post(url, data=json.dumps(service_data)).json()
            respond_result = respond['respond_result']
            
            # predict_sr_img = decodeRespond(respond_result['sr_out'])
            predict_compress_img = base64toPIL(respond_result['compress_sr_out'])
            
            # cv2.imencode('.jpg', predict_sr_img)[1].tofile(osp.join(save_sr_dir, osp.splitext(file)[0] + '.jpg'))
           
            # predict_compress_img = np.array(predict_compress_img)
            # predict_compress_img = cv2.cvtColor(predict_compress_img, cv2.COLOR_RGB2BGR)
            # cv2.imencode('.jpg', predict_compress_img)[1].tofile(osp.join(save_compress_dir, osp.splitext(file)[0] + '.jpg'))

            predict_compress_img.save(osp.join(save_compress_dir, osp.splitext(file)[0] + '.jpg'))
