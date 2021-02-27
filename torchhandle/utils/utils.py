import random
import os
import numpy as np
import torch
def seed_everything(seed,benchmark=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # the following line gives ~10% speedup
    # but may lead to some stochasticity in the results
    torch.backends.cudnn.benchmark = benchmark

def get_loader(ds,**kwargs):
    return torch.utils.data.DataLoader(ds,**kwargs)
