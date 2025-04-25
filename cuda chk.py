import torch

import sklearn
print(sklearn.__version__)

def list_cuda_gpus():
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA-capable GPUs: {num_gpus}\n")

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert bytes to GB
        compute_capability = torch.cuda.get_device_properties(i).major, torch.cuda.get_device_properties(i).minor
        print(f"GPU {i}: {gpu_name}")
        print(f"    Memory: {gpu_memory:.2f} GB")
        print(f"    Compute Capability: {compute_capability[0]}.{compute_capability[1]}\n")


if __name__ == "__main__":
    list_cuda_gpus()

print(torch.cuda.is_available())  # should be True
print(torch.version.cuda)         # e.g., '11.8'
print(torch.backends.cudnn.enabled)  # True if cuDNN is enabled
print(torch.distributed.is_nccl_available())

import torch
import gc
import multiprocessing as mp


def check_memory():
    """Prints allocated and reserved memory for each GPU."""
    for i in range(torch.cuda.device_count()):
        print(
            f"GPU {i}: {torch.cuda.memory_allocated(i)} bytes allocated, {torch.cuda.memory_reserved(i)} bytes reserved")


def clear_memory():
    """Attempts to release as much GPU memory as possible."""
    print("Before cleanup:")
    check_memory()

    # Garbage collection
    gc.collect()

    # Empty CUDA cache
    torch.cuda.empty_cache()

    # IPC collection for multiprocessing
    torch.cuda.ipc_collect()

    # Reset memory tracking
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()

    # Optional: Reset CUDA context (WARNING: Kills all active CUDA states)
    # torch.cuda.reset()

    print("After cleanup:")
    check_memory()


def main():
    """Main function to ensure proper memory cleanup."""
    if torch.cuda.is_available():
        print(
            f"Using {torch.cuda.device_count()} GPU(s): {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
        clear_memory()
    else:
        print("CUDA is not available. No cleanup needed.")


if __name__ == "__main__":
    # Ensure safe multiprocessing in some environments
    mp.set_start_method('spawn', force=True)
    main()
