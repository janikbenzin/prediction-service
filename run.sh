#!/bin/bash
ORT_DYLIB_PATH=$HOME/.local/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.so
echo $ORT_DYLIB_PATH
cd ./target/debug
./rust-onnx
