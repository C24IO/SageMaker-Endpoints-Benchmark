import mxnet as mx
import os
import json
import numpy as np
from collections import namedtuple 

ctx = mx.cpu()
dtype='float32'
Batch = namedtuple('Batch', ['data'])

mod = None
def model_fn(model_dir):
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
    img = mx.img.imdecode(img)
    img = mx.img.imresize(img, 224, 224)
    img = img.transpose((2, 0, 1))
    img = img.expand_dims(axis=0)
    mod.forward(Batch([img]))
    prob = mod.get_outputs()[0].asnumpy().tolist()
    prob_json = json.dumps(prob)
    return prob_json