import random
import numpy as np
import torch

def set_random_seed(seed):
    random.seed(seed)                      # Python random
    np.random.seed(seed)                   # Numpy random
    torch.manual_seed(seed)                 # Torch CPU random
    torch.cuda.manual_seed(seed)            # Torch current GPU random
    torch.cuda.manual_seed_all(seed)        # Torch all GPUs (if you use DataParallel or DDP)
    torch.backends.cudnn.deterministic = True    # Force CuDNN to be deterministic
    torch.backends.cudnn.benchmark = False       # Disable CuDNN auto-tuner (it can introduce randomness)