import os
import torch
import logging

def file_exists(folder_path, prefix="0.01"):
    # Look for the file whose name starts with the prefix
    for file_name in os.listdir(folder_path):
        if file_name.startswith(prefix):
            file_path = os.path.join(folder_path, file_name)
            # Ensure it is a file (not a directory)
            if os.path.isfile(file_path):
                return True

    return False


def select_device():
    # Check the total number of GPUs
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        return torch.device("cpu")

    # Get memory usage for each GPU
    min_usage = float('inf')
    best_device = None

    for i in range(num_gpus):
        mem_usage = torch.cuda.memory_allocated(i)
        print(f"GPU {i} memory usage: {mem_usage}")

        if mem_usage < min_usage:
            min_usage = mem_usage
            best_device = torch.device(f"cuda:{i}")

    return best_device


def set_logger(log_path):
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(message)s',
        level=logging.INFO
    )

    return logging.getLogger(log_path)