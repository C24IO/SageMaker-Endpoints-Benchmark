#!/usr/bin/env python

import mxnet as mx

print(mx.test_utils.list_gpus())

print(mx.gpu())

a = mx.nd.ones((2, 3), mx.gpu())
b = a * 2 + 1
print(b)