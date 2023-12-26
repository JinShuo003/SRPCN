from torch import nn
from torch.autograd import Function
import torch
import importlib
import os
chamfer_found = importlib.find_loader("chamfer_3D") is not None
if not chamfer_found:
    ## Cool trick from https://github.com/chrdiller
    print("Jitting Chamfer 3D")

    from torch.utils.cpp_extension import load
    chamfer_3D = load(name="chamfer_3D",
          sources=[
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
              "/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer3D.cu"]),
              ])
    print("Loaded JIT 3D CUDA chamfer distance")

else:
    import chamfer_3D
    print("Loaded compiled 3D CUDA chamfer distance")


# Chamfer's distance module @thibaultgroueix
# GPU tensors only


