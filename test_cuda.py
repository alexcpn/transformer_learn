import torch
import torchvision

print("torch version is ",torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


x=torch.ones(1,3,224,224).to(device)
model=torchvision.models.resnet50().to(device)
compiled=torch.compile(model)
compiled(x)


"""
https://leimao.github.io/blog/CUDA-Driver-VS-CUDA-Runtime/

libcuda.so is installed via the driver at

/usr/lib/x86_64-linux-gnu/libcuda.so.525.105.17
/usr/lib/x86_64-linux-gnu/libcuda.so.1

CUDA Runtime installs libcudart.so at
ld  -L/usr/local/cuda/lib64/ -lcudart --verbose

attempt to open /usr/local/cuda/lib64//libcudart.so succeeded
/usr/local/cuda/lib64//libcudart.so

"""

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

