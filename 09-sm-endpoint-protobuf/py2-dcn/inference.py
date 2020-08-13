#LOCAL inference.py - EDIT THIS

"""======CUSTOM============"""
import sys
sys.path.append('/home/model-server')

"""======CUSTOM============"""
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

"""======CUSTOM============"""

print('~custom start')

from json import JSONEncoder
from dcn.rfcn.config.config import config, update_config
from dcn.lib.utils.image import resize, transform
from dcn.rfcn import _init_paths
import numpy as np
import argparse
import logging
import pprint
import sys
import cv2
import os
import mxnet as mx
from dcn.rfcn.core.tester import im_detect, Predictor
from dcn.rfcn.symbols import *
from dcn.lib.utils.load_model import load_param
#from dcn.lib.utils.show_boxes import show_boxes
from dcn.lib.utils.tictoc import tic, toc
from dcn.lib.nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
update_config('/home/model-server/dcn/experiments/rfcn/cfgs/rfcn_coco_demo.yaml')
sys.path.insert(0, os.path.join('/home/model-server/dcn/external/mxnet', config.MXNET_VERSION))
# config.symbol = 'resnet_v1_101_rfcn'
config.symbol = 'resnet_v1_101_rfcn_dcn'
sym_instance = eval(config.symbol + '.' + config.symbol)()
sym = sym_instance.get_symbol(config, is_train=False)

num_classes = 81
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
           'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
           'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

print('~custom end')

"""======CUSTOM============"""

logger = logging.getLogger()
logger.setLevel(logging.INFO)        

print('~inference.py')

"""
if(mx.context.num_gpus() > 0):
        pprint.pprint(config)
        a = mx.nd.ones((2, 3), mx.gpu())
        b = a * 2 + 1
        print(b)
"""

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Use GPU if one exists, else use CPU
ctx = mx.gpu()

dtype='float32'
Batch = namedtuple('Batch', ['data'])

arg_params = None 
aux_params = None

mod = None
def model_fn(model_dir):
    
    logger.info("\nmodel_fn\n")
    arg_params, aux_params = load_param('/.sagemaker/mms/models/model/rfcn_dcn_coco', 0, process=True)
        
    return mod

import image_pb2 as impb
image_packet = impb.PBImage()

def transform_fn(mod, img, input_content_type, output_content_type):
    
    logger.info("\ntransform_fn\n")
    
    data = []
    data_names = ['data', 'im_info']
    label_names = []

    #image_name = '/home/model-server/dcn/sv.jpg'
    #im = cv2.imread(image_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    
    if(input_content_type == 'application/octet-stream'):
        print(input_content_type)
        image_packet.ParseFromString(img)
        img = image_packet.image_data
        
    bytes_im = np.asarray(bytearray(img))
    print(type(img))
    
    im = cv2.imdecode(bytes_im, cv2.IMREAD_UNCHANGED)
    height, width, channels = im.shape
    print(str(height) + ' ' + str(width) + ' ' + str(channels))
    
    target_size = config.SCALES[0][0]
    max_size = config.SCALES[0][1]
    im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
    im_tensor = transform(im, config.network.PIXEL_MEANS)
    im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
    data.append({'data': im_tensor, 'im_info': im_info})

    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]

    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]

    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]

    #arg_params, aux_params = load_param('./dcn/model/rfcn_dcn_coco', 0, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)

    data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                 provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                 provide_label=[None])

    scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
    scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)

    idx=0
    #im_name=image_name

    data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                 provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                 provide_label=[None])
    scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

    scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
    boxes = boxes[0].astype('f')
    scores = scores[0].astype('f')
    dets_nms = []
    for j in range(1, scores.shape[1]):
        cls_scores = scores[:, j, np.newaxis]
        cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores))
        keep = nms(cls_dets)
        cls_dets = cls_dets[keep, :]
        cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
        dets_nms.append(cls_dets)
    
    #print(' '.join(map(str, scores)))
    # Serialization
    numpyData = {"array": dets_nms}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    return encodedNumpyData

def main():
    model_fn(None)
    transform_fn(None, None, None, None)

if __name__ == "__main__":
    # execute only if run as a script
    main()

