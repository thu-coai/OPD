import argparse
import torch
from collections import OrderedDict
from tqdm import tqdm


def merge(ckpt_path):
    param1 = torch.load(f"{ckpt_path}/checkpoint-001.pt", map_location='cpu')
    param2 = torch.load(f"{ckpt_path}/checkpoint-002.pt", map_location='cpu')
    param1.update(param2)
    
    torch.save(param1, f"{ckpt_path}/checkpoint.pt")
    

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_args()
    
    merge(args.ckpt_path)

