import torch.cuda as cuda


def print_cuda_alloc_mem():
    gb = cuda.memory_allocated() / 1024 / 1024 / 1024

    print(f"{round(gb,3)} GB allocated")

    return gb
