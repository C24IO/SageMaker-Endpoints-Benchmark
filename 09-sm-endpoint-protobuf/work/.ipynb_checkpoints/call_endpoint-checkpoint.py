#!/usr/bin/env python

import io
import numpy as np
import json
from sagemaker.predictor import StringDeserializer
import time
import image_pb2 as impb


with open('/tmp/test.jpg', 'rb') as f:
    payload = f.read()
    #payload = bytearray(payload)

image_packet = impb.Image()
image_packet.image_data = payload

def numpy_bytes_serializer(data):
    f = io.BytesIO()
    np.save(f, data)
    f.seek(0)
    return f.read()

predictor.serializer = None
predictor.deserializer = StringDeserializer()
predictor.accept = None
predictor.content_type = 'application/octet-stream'

for i in range(0, 10):    
    response = predictor.predict(image_packet.SerializeToString())            
    time.sleep(1)

#response