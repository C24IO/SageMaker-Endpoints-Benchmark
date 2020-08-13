#!/usr/bin/env python

import image_pb2 as impb
from PIL import Image
import io
import imutils
import cv2

# Read the existing image.
image_packet = impb.Image()

try:
  f = open("image_data.proto.bin", "rb")
  image_packet.ParseFromString(f.read())
  f.close()
except IOError:
  print("image_data.proto.bin" + ": Could not open file.")

with open('picture_out.jpg', 'wb') as f:
    f.write(image_packet.image_data)

image_transform = cv2.imread("picture_out.jpg")
(h, w, d) = image_transform.shape
print("width={}, height={}, depth={}".format(w, h, d))
(B, G, R) = image_transform[100, 50]
print("R={}, G={}, B={}".format(R, G, B))
roi = image_transform[60:160, 320:420]
resized = cv2.resize(image_transform, (100, 100))
rotated = imutils.rotate(image_transform, -45)


#with open('picture_out.jpg', 'wb') as imagefile:
    #imagefile.write(bytearray(image_packet.SerializeToString()))
#bytesIOImage = io.BytesIO(bytearray(image_packet.SerializeToString()))
#bytesIOImage.seek(0)
#image = Image.open(bytesIOImage)
#image.save('picture_out.jpg')

