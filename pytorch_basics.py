import torch
import numpy as np

# numpy -> torch tensors
np_arr = np.ones((2,2))
print(np_arr)

torch_tensor = torch.from_numpy(np_arr)
print(torch_tensor)

# torch tensor -> numpy
tensor_ones = torch.ones(2,2)
print(tensor_ones)

np_arr = tensor_ones.numpy()
print(np_arr)

# tensors on CPU and GPU
tensor_cpu = torch.ones(2,2)

if torch.cuda.is_available():
    print("cuda is available!")
    tensor_cpu.cuda()   # toggle variable to be in gpu
tensor_cpu.cpu()        # toggle variable to be in cpu

# basic operations
a = torch.ones(2,2)
b = 2 * torch.ones(2,2)

c = a + b
print(c)

c = torch.add(a, b)
print(c)

print(a.add_(b)) # done in-place addition. any _ function is in-place
print(a)

rand_tensor = torch.rand(3,3)
print(rand_tensor)

print(rand_tensor.mean(dim=0))
print(rand_tensor.std(dim=0))

