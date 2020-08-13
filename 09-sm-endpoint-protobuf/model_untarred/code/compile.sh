#!/usr/bin/env bash

#/Users/chazarey/Downloads/protoc-3.12.1-osx-x86_64/bin/protoc --python_out=./proto.out addressbook.proto
#/Users/chazarey/Downloads/protoc-3.12.1-osx-x86_64/bin/protoc --python_out=./proto.out image.proto
protoc --python_out=./ image.proto

