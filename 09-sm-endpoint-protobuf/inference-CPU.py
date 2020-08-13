#LOCAL inference.py - EDIT THIS

import mxnet as mx
import os
import json
import numpy as np
from collections import namedtuple
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import image_pb2 as impb
image_packet = impb.PBImage()
        

'''
import os.path
from os import path
with open('/tmp/ttarred.flg', 'w') as fp: 
    pass

#untar protobuf files
import tarfile
proto_tar = tarfile.open('protobuf.tar.gz')
proto_tar.extractall('/tmp/')
proto_tar.close()
'''

print('~')

#ctx = mx.gpu()
ctx = mx.cpu()

dtype='float32'
Batch = namedtuple('Batch', ['data'])

mod = None
def model_fn(model_dir):
    
    logger.info("model_fn")
    
    
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    for arg in arg_params:
        arg_params[arg] = arg_params[arg].astype(dtype).as_in_context(ctx)

    for arg in aux_params:
        aux_params[arg] = aux_params[arg].astype(dtype).as_in_context(ctx)

    exe = mod.bind(for_training=False,
               data_shapes=[('data', (1,3,224,224))],
               label_shapes=mod._label_shapes)

    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod


def transform_fn(mod, img, input_content_type, output_content_type):
    
    logger.info("transform_fn")
    
    if(input_content_type == 'application/octet-stream'):
        print(input_content_type)
        image_packet.ParseFromString(img)
        img = mx.img.imdecode(image_packet.image_data)
        img = mx.img.imresize(img, 224, 224)
        img = img.transpose((2, 0, 1))
        img = img.expand_dims(axis=0)
        mod.forward(Batch([img]))
        prob = mod.get_outputs()[0].asnumpy().tolist()
        prob_json = json.dumps(prob)
        return prob_json
    
    img = mx.img.imdecode(img)
    img = mx.img.imresize(img, 224, 224)
    img = img.transpose((2, 0, 1))
    img = img.expand_dims(axis=0)
    mod.forward(Batch([img]))
    prob = mod.get_outputs()[0].asnumpy().tolist()
    prob_json = json.dumps(prob)
    return prob_json