import torch
import time
def print_t(tensor):
    print(f"shape of tensor {tensor.shape}")
    print(f"type of tensor {tensor.dtype}")
    print(f"device of tensor {tensor.device}")


tensor=torch.randn([3000,4000])
tensor2=torch.randn([4000,3000])
start=time.time()
for i in range(100):
    res = torch.matmul(tensor, tensor2)
print(f"{time.time()-start}")





