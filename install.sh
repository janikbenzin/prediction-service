sudo dnf install rocm
sudo usermod -aG render,video $USER
rocminfo | grep gfx1103
sudo dnf install onnxruntime-rocm onnxruntime-rocm-devel
sudo ln -sf /usr/lib64/rocm/lib/libonnxruntime.so.1.20.1 /usr/lib64/libonnxruntime.so
sudo ln -sf /usr/lib64/rocm/lib/libonnxruntime.so.1.20.1 /usr/lib64/libonnxruntime.so.1