import torch

def set_device(name=None, verbose=True):
    if name is None:
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(name)
        
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

        if verbose:
            device_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_id)
            gpu_properties = torch.cuda.get_device_properties(device_id)
            total_memory = gpu_properties.total_memory / (1024 ** 3)  # Convert bytes to GB

            print(f"GPU: {gpu_name}", flush=True)
            print(f"Total Memory: {total_memory:.2f} GB", flush=True)
    else:
        if verbose:
            print("No GPU available. Using CPU instead.", flush=True)



def get_device():
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        device_id = torch.cuda.current_device()
        return torch.device(f"cuda:{device_id}")
    else:
        return torch.device("cpu")
    