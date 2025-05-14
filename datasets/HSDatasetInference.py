import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset as TorchvisionDataset

from datasets.SkinPatchDataset import SkinPatchDataset
from datasets.SL1HSDataset import MultiSubjectSL1HSDBDataset

from utils.preprocessing import global_contrast_normalization

class HSDatasetInference(TorchvisionDataset):
    def __init__(self, 
                 patch_size=32,
                 applyTransform=True):
        
        super().__init__(root=None)  # No root directory needed for on-the-fly generation
        
        self.applyTransform = applyTransform
        self.patch_size = patch_size

        # Pre-computed min and max values (after applying GCN)  
        min_value, max_value = (-2.0743157863616943, 3.0839202404022217) # data from 10 sub 10 patches/sub
        
        if self.applyTransform:
            # Transformations for the dataset
            transform = transforms.Compose([
                    transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                    transforms.Normalize([min_value] * 31, [max_value - min_value] * 31)
                ])
            print("Transformations applied")
        else:
            transform = None
            print("No transformations applied")

        # Noise scale for the dataset
        noise_scale = 0.025

        # -------------------------------------------------
        # Train set: Only real patches
        # -------------------------------------------------

        self.train_set = SkinPatchDataset(
            num_subjects=100,
            patches_per_subject=10,
            patch_size=patch_size,
            noise_scale=noise_scale,
            applyRandomIllumination=False,
            isRealSkin=True,  # Only real patches
            transform=transform,
        )
        
        # -------------------------------------------------
        # Test set: Real and fake patches
        # -------------------------------------------------

        part_1 = MultiSubjectSL1HSDBDataset(
            num_subjects=50,
            patches_per_file= 2, # num_real_patches // num_subjects,
            patch_size=patch_size,
            transform=transform,
        )
    
        # Fake patches for the test set
        part_2 = SkinPatchDataset(
            num_subjects=100,
            patches_per_subject= 1, 
            patch_size=patch_size,
            noise_scale=noise_scale,
            isRealSkin=False,  # Fake patches
            applyRandomIllumination=True,
            transform=transform,
        )

        # Set global offsets for test sets
        part_1.global_offset = 0
        part_2.global_offset = len(part_1) 

        # Combine real and fake test sets and adjust indices
        self.test_set = torch.utils.data.ConcatDataset([
            part_1,
            part_2
        ])  