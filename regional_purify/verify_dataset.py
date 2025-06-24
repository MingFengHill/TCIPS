import numpy as np
import random
import os
import h5py
import argparse

def print_items(group, indent=0):
    indent_str = '    ' * indent
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            print(f"{indent_str}Dataset '{key}' - Shape: {item.shape}, Type: {item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"{indent_str}Group '{key}'")
            print_items(item, indent + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, default='/home/rh/data/data_parsenet/train_data.h5')
    # parser.add_argument('--data_path', type=str, default='/home/rh/data/data_save/train_data_30_3ee0.h5')
    parser.add_argument('--data_path', type=str, default='/home/rh/data/data_save/train_data_sed_30_ec89.h5')
    # parser.add_argument('--data_path', type=str, default='/home/rh/data/data/train_data_withEdge.h5')
    args = parser.parse_args()
    for arg, value in sorted(vars(args).items()):
        print("[INFO] Argument {}: {}".format(arg, value))

    with h5py.File(args.data_path, 'r') as file:
        print_items(file)