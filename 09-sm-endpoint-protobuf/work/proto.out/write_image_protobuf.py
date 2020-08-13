#!/usr/bin/env python

import image_pb2 as impb
import sys
import os
import numpy

with open('008_0007.jpg', 'rb') as f:
    payload = f.read()
    #payload = bytearray(payload)

image_packet = impb.Image()
image_packet.image_data = payload

# Write the new address book back to disk.
f = open("image_data.proto.bin", "wb")
f.write(image_packet.SerializeToString())
f.close()


