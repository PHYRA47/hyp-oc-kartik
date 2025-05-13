import os
import random
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


class SL1HSDBDataset(Dataset):
    def __init__(self, 
                 file_idx, 
                 patches_per_file=1, 
                 patch_size=32, 
                 interpolate=True, 
                 overlapping=False, 
                 verbose=False):
        """
        Base class for generating patches from a single .mat file.

        Args:
            file_idx (int): Index of the .mat file to process.
            patches_per_file (int): Number of patches to generate per file.
            patch_size (int): Size of the square patch (height and width).
            interpolate (bool): Whether to interpolate the reflectance data to target wavelengths.
            overlapping (bool): Whether to allow overlapping patches.
            verbose (bool): Whether to print debug information.
        """
        self.path = '/cig/common02nb/HS_skin_DB/RGB2HSI_dataset/2_Dataset/SL1DB_RGB2HSI_raw_HSI/HS_cubes/'
        self.file_idx = file_idx
        self.patches_per_file = patches_per_file
        self.patch_size = patch_size
        self.interpolate = interpolate
        self.overlapping = overlapping
        self.verbose = verbose

        # Load all .mat files in the specified path and sort them
        self.mat_files = sorted([f for f in os.listdir(self.path) if f.endswith('.mat')])

        # Ensure the file index is valid
        if self.file_idx >= len(self.mat_files):
            raise IndexError(f"File index {self.file_idx} is out of range (0-{len(self.mat_files)- 1}).")

        # Define target wavelengths for interpolation if needed
        if self.interpolate:
            self.target_wavelengths = np.linspace(400, 700, 31)

        if self.verbose: # {file_name[11:-16]:<25}
            print(f"{'Initializing SL1HSDB'.center(42, '=')}")
            print(f"{('.mat file: ' + self.mat_files[self.file_idx][11:-16]).center(42, '-')}")
            print(f"{'File index':<30}: {self.file_idx:>10}")
            print(f"{'Patches per file':<30}: {self.patches_per_file:>10}")
            print(f"{'Patch size':<30}: {f'{self.patch_size} x {self.patch_size}':>10}")
            print(f"{'Overlapping patches':<30}: {self.overlapping:>10}")
            print(f"Initialization Complete".center(42, '='))
            print('\n')

    def _load_reflectance(self):
        """
        Load reflectance data from the specified .mat file.

        Returns:
            np.ndarray: Reflectance data.
        """
        file_name = self.mat_files[self.file_idx]
        file_path = os.path.join(self.path, file_name)
        mat_data = scipy.io.loadmat(file_path)
        hsi_data = mat_data['HS_skin_data']
        reflectance = hsi_data['reflectance'][0][0]

        if self.interpolate:
            original_wavelengths = hsi_data['wavelengths'][0][0][0]
            reflectance_interpolated = np.zeros((31, reflectance.shape[1], reflectance.shape[2]))
            for i in range(reflectance.shape[1]):  # Iterate over rows
                for j in range(reflectance.shape[2]):  # Iterate over columns
                    reflectance_interpolated[:, i, j] = np.interp(
                        self.target_wavelengths, original_wavelengths, reflectance[:, i, j]
                    )
            reflectance = reflectance_interpolated
        
        return reflectance

    def _generate_patches(self, reflectance):
        """
        Generate patches from the reflectance data.

        Args:
            reflectance (np.ndarray): Reflectance data.

        Returns:
            list of torch.Tensor: List of patches as tensors.
        """
        stride = self.patch_size // 2 if self.overlapping else self.patch_size
        h, w = reflectance.shape[1], reflectance.shape[2]

        # Generate all possible patch positions
        possible_positions = [
            (i, j)
            for i in range(0, h - self.patch_size + 1, stride)
            for j in range(0, w - self.patch_size + 1, stride)
        ]

        # Randomly sample positions if patches_per_file is specified
        num_patches = min(self.patches_per_file, len(possible_positions))
        selected_positions = random.sample(possible_positions, num_patches) # randomly select positions

        patches = []
        for i, j in selected_positions:
            patch = reflectance[:, i:i + self.patch_size, j:j + self.patch_size]
            patches.append((torch.tensor(patch, dtype=torch.float32), (i, j)))

        return patches

    def __len__(self):
        return self.patches_per_file

    def __getitem__(self, idx):
        """
        Get a patch from the dataset.

        Args:
            idx (int): Index of the patch.

        Returns:
            tuple: (patch_tensor, coordinates)
        """
        reflectance = self._load_reflectance()
        patches = self._generate_patches(reflectance)

        # Ensure we don't exceed the number of available patches
        patch_tensor, coordinates = patches[idx % len(patches)]

        if self.verbose:
            if idx == 0:
                print(f"Generating patches from {self.mat_files[self.file_idx][11:-16]}".center(60, ' '))
                print(f"{'-' * 60}")
                print(f"{'Index':<20}{'Patch Number':<20}{'Coordinates':<20}")
                print(f"{'-' * 60}")
            print(f"{idx:<20}{idx:<20}{str(coordinates):<20}")

        return patch_tensor, coordinates


class MultiSubjectSL1HSDBDataset(SL1HSDBDataset):
    def __init__(self, num_subjects=70, randomize=True, patches_per_file=1, transform=None, verbose=False, **kwargs):
        """
        Derived class for generating patches from multiple .mat files.

        Args:
            num_subjects (int): Number of subjects to process.
            randomize (bool): Whether to randomly select the files.
            patches_per_file (int): Number of patches to generate per file.
            kwargs: Additional arguments for the base class.
        """
        super().__init__(file_idx=0, patches_per_file=patches_per_file, **kwargs)
        self.num_subjects = num_subjects
        self.randomize = randomize
        self.transform = transform
        self.verbose = verbose

        # Select files based on num_subjects and randomize
        if self.randomize:
            self.selected_files = random.sample(self.mat_files, min(self.num_subjects, len(self.mat_files)))
        else:
            self.selected_files = self.mat_files[:min(self.num_subjects, len(self.mat_files))]

        if self.verbose:
            print(f"{'Initializing MultiSL1HSDB'.center(42, '=')}")
            print(f"{'Number of subjects':<30}: {len(self.selected_files):>10}")
            print(f"{'Patches per file':<30}: {self.patches_per_file:>10}")
            print(f"{'Total patches':<30}: {self.num_subjects * self.patches_per_file:>10}")
            print(f"{'Patch size':<30}: {f'{self.patch_size} x {self.patch_size}':>10}")
            print(f"Initialization Complete".center(42, '='))
            print('\n')

    def __len__(self):
        return len(self.selected_files) * self.patches_per_file

    def __getitem__(self, idx):
        """
        Get a patch from the dataset.

        Args:
            idx (int): Global index of the patch.

        Returns:
            tuple: (patch_tensor, label, global_idx)
        """
        file_idx = idx // self.patches_per_file
        patch_idx = idx % self.patches_per_file

        file_name = self.selected_files[file_idx]
        self.file_idx = self.mat_files.index(file_name)  # Update file_idx for the base class
        reflectance = self._load_reflectance()
        patches = self._generate_patches(reflectance)

        # Ensure we don't exceed the number of available patches
        patch_tensor, coordinates = patches[patch_idx % len(patches)]

        if self.transform:
            patch_tensor = self.transform(patch_tensor)

        label = 0  # Label indicating real skin
        
        # Convert coordinates to global index
        if hasattr(self, 'global_offset'):
            global_idx = idx + self.global_offset
        else:
            global_idx = idx

        if self.verbose:
            if idx == 0:  # Print the header only once when the method is called
                print(f"Generating patches from {len(self.selected_files)} .mat files".center(60, ' '))
                print(f"{'-' * 60}")
                print(f"{'Index':<10}{'.mat File':<25}{'Patch No.':<10}{'Coordinates':<15}")
                print(f"{'-' * 60}")
            print(f"{idx:<10}{file_name[11:-16]:<25}{patch_idx:<10}{str(coordinates):<15}")

        return patch_tensor, label, global_idx