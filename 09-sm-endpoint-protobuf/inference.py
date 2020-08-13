#LOCAL inference.py - EDIT THIS

import mxnet as mx
import os
import json
import numpy as np
from collections import namedtuple
import logging
import cv2
import PIL
from PIL import Image
import io

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import image_pb2 as impb
image_packet = impb.PBImage()
        

print('~inference.py')

# Use GPU if one exists, else use CPU
ctx = mx.gpu()

dtype='float32'
Batch = namedtuple('Batch', ['data'])

mod = None
def model_fn(model_dir):
    
    logger.info("\nmodel_fn\n")
    
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)    
    
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    
    return mod


def transform_fn(mod, img, input_content_type, output_content_type):
    
    logger.info("\ntransform_fn\n")
    
    prob_json = json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])     
    return prob_json

    if(input_content_type == 'application/octet-stream'):
        print(input_content_type)       
        image_packet.ParseFromString(img)
        img = np.array(Image.open(io.BytesIO(image_packet.image_data))) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]
        mod.forward(Batch([mx.nd.array(img)]))
        prob = mod.get_outputs()[0].asnumpy().tolist()
        prob_json = json.dumps(prob)      
        return prob_json
    
    print(input_content_type)
    img = np.array(Image.open(io.BytesIO(img))) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy().tolist()
    prob_json = json.dumps(prob)      
    return prob_json
