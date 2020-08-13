#!/usr/bin/env python

from flask import Flask
from flask import request
import image_pb2 as impb
from PIL import Image
import io
import imutils
import cv2

app = Flask(__name__)

@app.route('/invocations', methods = ['POST'])
def index():
    data = request.data
    # Read the existing image.
    image_packet = impb.Image()
    image_packet.ParseFromString(data)
    with open('picture_out_for.jpg', 'wb') as f:
        f.write(image_packet.image_data)
        f.close()

    img = Image.open('picture_out_for.jpg')

    # resize the image
    width, height = img.size
    aspect_ratio = height / width
    new_width = 120
    new_height = aspect_ratio * new_width * 0.55
    img = img.resize((new_width, int(new_height)))
    # new size of image
    # print(img.size)

    # convert image to greyscale format
    img = img.convert('L')

    pixels = img.getdata()

    # replace each pixel with a character from array
    chars = ["B", "S", "#", "&", "@", "$", "%", "*", "!", ":", "."]
    new_pixels = [chars[pixel // 25] for pixel in pixels]
    new_pixels = ''.join(new_pixels)

    # split string of chars into multiple strings of length equal to new width and create a list
    new_pixels_count = len(new_pixels)
    ascii_image = [new_pixels[index:index + new_width] for index in range(0, new_pixels_count, new_width)]
    ascii_image = "\n".join(ascii_image)
    print(ascii_image)

    # write to a text file.
    with open("ascii_image.txt", "w") as f:
        f.write(ascii_image)
        f.close()

    with open("ascii_image.txt", 'r') as f:
        file_contents = f.read()
        print(file_contents)
        f.close()

    return(file_contents)

if __name__ == '__main__':
    app.run(debug=True)

