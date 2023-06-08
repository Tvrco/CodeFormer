import os,sys
import json
import base64
import numpy as np
from flask import Flask,Response,request
from sr_model import Img_SR_Model
from basic import *
import cv2
app = Flask(__name__)
import logging

model_config = 'config.yaml'
SR_Model = Img_SR_Model(model_config)

def imgtostr(respond, ext='.png'):
    imgstr = cv2.imencode(ext, respond)[1]
    imgbyte = imgstr.tobytes()
    imgb64 = base64.b64encode(imgbyte)
    imgb64 = imgb64.decode()  # str
    respond = imgb64
    return respond

def parseHttpRequest(request):
    try:
        str_base64 = request['image']
        imgtype = request['poster_type']
    except:
        raise DataParseError()
    if imgtype not in ['small', 'large']:
        raise DataTypeError()
    try:
        data_decode = str_base64.encode('ascii')  # byte
        byte_base64 = base64.b64decode(data_decode)
        image = np.asarray(bytearray(byte_base64), dtype='uint8')  # ndarray
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image, imgtype
    except:
        raise DataParseError()

@app.route("/runImageSR",methods=['POST'])
def runImageSR():
    try:
        _request = json.loads(request.get_data().decode('utf-8'))
        print('reqest data sucess!!!!!!')
        img, imgtype = parseHttpRequest(_request)
        returnData = SR_Model.inference(img, imgtype)
        print('Inference finished!!!!')
        
        # returnData['sr_out'] = imgtostr(returnData['sr_out'])
        # returnData['compress'] = imgtostr(returnData['compress'])
        
        Msg = {'respond_code':200, 'respond_message':'Success', 'respond_result':returnData}

    except DataParseError as e:
        logging.exception(f"main exception: {str(e)}")
        Msg = {'respond_code': e.code, 'respond_message': e.message}

    except Exception as e:
        logging.exception(f"main exception: {str(e)}")
        Msg = {'respond_code':210,'respond_message':str(e)}  # 其他错误

    return json.dumps(Msg, ensure_ascii=False), 200, {'content-type':'application/json', 'X-Frame-Options':'DENY'}

