import image_pb2 as impb

with open('008_0007.jpg', 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)

