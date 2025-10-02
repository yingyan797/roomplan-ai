import json, os, random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class GridFileProcessor:
    def __init__(self, fname, n_categories=6):
        self.fname = fname
        self.n_categories = n_categories

    def _load_grid_file(self):
        with open(self.fname, "r") as f:
            self.content = json.load(f)
        shape = tuple(self.content["dim"])
        aug_grid = np.zeros(shape, dtype=np.bool_)
        empty_grid = np.zeros(shape+(self.n_categories,), dtype=np.bool_)
        empty_grid[:, :, 0] = 1
        for layer in self.content["layers"]:
            i = layer["id"]+1
            if layer["data"]:
                coords = np.array([list(map(int, k.split(","))) for k in layer["data"].keys()], dtype=np.int_)
                rows, cols = coords[:, 0], coords[:, 1]
                if i != 2:
                    empty_grid[rows, cols] = 0
                    empty_grid[rows, cols, i] = 1
                else:
                    aug_grid[rows, cols] = 1
                    self.placements = coords
        self.aug_grid = aug_grid
        self.empty_grid = empty_grid

    def create_input_output(self, n_partial=1):
        self._load_grid_file()
        
        inputs, outputs = [self.empty_grid], [self.aug_grid]
        if len(self.placements) > 1:
            for _ in range(n_partial):
                n_select = random.randint(1, len(self.placements)-1)
                selected = np.array(random.sample(self.placements.tolist(), n_select), dtype=np.int_)
                
                partial_input = self.empty_grid.copy()
                rows, cols = selected[:, 0], selected[:, 1]
                partial_input[rows, cols] = 0
                partial_input[rows, cols, 2] = 1
                
                partial_output = self.aug_grid.copy()
                partial_input[rows, cols] = 0
                inputs.append(partial_input)
                outputs.append(partial_output)
        return inputs, outputs

class GridDataset(Dataset):
    def __init__(self, grids, targets, augment=False):
        """
        Args:
            grids: List of 3D arrays [H, W, C=6] with one-hot encoded categories
            targets: List of 2D arrays [H, W] with binary labels
            augment: Whether to apply data augmentation
        """
        self.grids = grids
        self.targets = targets
        self.augment = augment
        
    def __len__(self):
        return len(self.grids)
    
    def _pad_to_divisible(self, arr, divisor=8):
        """Pad array to make dimensions divisible by divisor"""
        if arr.ndim == 3:  # [H, W, C]
            h, w, c = arr.shape
            new_h = ((h + divisor - 1) // divisor) * divisor
            new_w = ((w + divisor - 1) // divisor) * divisor
            
            if new_h == h and new_w == w:
                return arr
            
            padded = np.zeros((new_h, new_w, c), dtype=arr.dtype)
            padded[:h, :w, :] = arr
        else:  # [H, W]
            h, w = arr.shape
            new_h = ((h + divisor - 1) // divisor) * divisor
            new_w = ((w + divisor - 1) // divisor) * divisor
            
            if new_h == h and new_w == w:
                return arr
            
            padded = np.zeros((new_h, new_w), dtype=arr.dtype)
            padded[:h, :w] = arr
        
        return padded
    
    def __getitem__(self, idx):
        grid = self.grids[idx].copy()  # [H, W, 6]
        target = self.targets[idx].copy()  # [H, W]
        
        if self.augment:
            # Random flip
            if np.random.rand() > 0.5:
                grid = np.flip(grid, axis=0).copy()
                target = np.flip(target, axis=0).copy()
            if np.random.rand() > 0.5:
                grid = np.flip(grid, axis=1).copy()
                target = np.flip(target, axis=1).copy()
            
            # Random rotation (90 degree increments)
            k = np.random.randint(0, 4)
            grid = np.rot90(grid, k, axes=(0, 1)).copy()
            target = np.rot90(target, k).copy()
        
        # Pad to make dimensions divisible by 8 for downsampling
        grid = self._pad_to_divisible(grid)
        target = self._pad_to_divisible(target)
        
        # Convert to [C, H, W] format for PyTorch
        grid = grid.transpose(2, 0, 1).astype(np.float32)  # [6, H, W]
        target = target.astype(np.float32)[np.newaxis, ...]  # [1, H, W]
        
        return torch.from_numpy(grid), torch.from_numpy(target)


def collate_variable_size(batch):
    """Custom collate function to handle variable-sized grids"""
    # Find max dimensions in batch
    max_h = max([item[0].shape[1] for item in batch])
    max_w = max([item[0].shape[2] for item in batch])
    
    # Pad all samples to max size
    padded_grids = []
    padded_targets = []
    
    for grid, target in batch:
        c, h, w = grid.shape
        
        # Pad grid
        padded_grid = torch.zeros(c, max_h, max_w)
        padded_grid[:, :h, :w] = grid
        padded_grids.append(padded_grid)
        
        # Pad target
        padded_target = torch.zeros(1, max_h, max_w)
        padded_target[:, :h, :w] = target
        padded_targets.append(padded_target)
    
    return torch.stack(padded_grids), torch.stack(padded_targets)

if __name__ == "__main__":
    gfp = GridFileProcessor("dataset/training_4.json")
    gfp._load_grid_file()
    print(gfp.create_input_output())
            
         