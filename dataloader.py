import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.interpolate import interp1d

from utils.RandomIllumination import RandomIllumination

class SkinPatchDataset(Dataset):
    
    def __init__(self, 
                 num_subjects=10, 
                 randomize=True,
                 patches_per_subject=10, 
                 patch_size=16, 

                 isRealSkin=True,
                 applyRandomIllumination=True,
                 applyNoise=True, noise_scale=0.025,
                 
                 transform=None, 
                 verbose=False):
        """
        Initialize the SkinPatchDataset.

        Args:
            num_subjects (int): Number of subjects to load.
            patches_per_subject (int): Number of patches per subject.
            patch_size (int): Size of each patch (square).
            noise_scale (float): Scale of the noise to add.
            isRealSkin (bool): Whether to load real or fake skin data.
            transform (callable, optional): Transform to apply to the patches.
            target_transform (callable, optional): Transform to apply to the labels.
            verbose (bool): If True, print debug information.
        """
        self.verbose = verbose

        if self.verbose:
            print(f"{'Initializing SkinPatchDataset'.center(42, '=')}")
            print(f"{'Real Skin':<30}: {str(isRealSkin):>10}")
            print(f"{'Randome Illumination':<30}: {str(applyRandomIllumination):>10}")
            print(f"{'Number of subjects':<30}: {num_subjects:>10}")
            print(f"{'Patches per subject':<30}: {patches_per_subject:>10}")
            print(f"{'Total patches':<30}: {num_subjects * patches_per_subject:>10}")
            print(f"{'Patch size':<30}: {f'{patch_size} x {patch_size}':>10}")
            # print("-"*42)

        # Dataset parameters
        self.num_subjects = num_subjects
        self.randomize = randomize
        self.patches_per_subject = patches_per_subject
        self.patch_size = patch_size
        self.total_patches = num_subjects * patches_per_subject
        
        self.isRealSkin = isRealSkin
        self.applyRandomIllumination = applyRandomIllumination
        self.applyNoise = applyNoise

        # File paths
        R_real_csv_path = [
            '/cig/common05nb/students/denegasf/datasets/1832_Data_JResNIST_skinrefl_v3_average_only.csv'
        ]
        R_fake_csv_path = [
            "/cig/common05nb/students/denegasf/datasets/UMINHO-HSFD/reconstructed/mst-plus-plus/reconstruction_reflectance_data.csv",
            "/cig/common05nb/students/denegasf/datasets/UMINHO-HSFD/reconstructed/restormer/reconstruction_reflectance_data.csv"
        ]
        sr_mat_path = '/cig/common05nb/students/denegasf/datasets/UMINHO-HSFD/grid_files_v2/combined_grid_stats.mat'

        # Load and process reflectance data based on isRealSkin
        if self.verbose:
            print(f"{'Loading ' + ('Real' if isRealSkin else 'Fake') + ' skin reflectance data'}".center(42, '-'))
        if isRealSkin:
            self.reflectance_data = self._load_reflectance(R_real_csv_path, interpolate=True, num_subjects=num_subjects)
        else:
            self.reflectance_data = self._load_reflectance(R_fake_csv_path, interpolate=False, num_subjects=num_subjects)

        # Load standard deviation data
        self.sr = loadmat(sr_mat_path)['std_of_mean_reflectance'][4][:31]

        # Initialize random illumination
        self.random_illum = RandomIllumination(p=0.7)  # Probability of applying random illumination
        
        # Sensor sensitivity (identity matrix for simplicity)
        self.sensor_sens = np.eye(31)

        # Noise scale
        self.noise_scale = noise_scale

        # Transformations
        self.transform = transform

        if self.verbose:
            print(f"Initialization Complete".center(42, '='))

    def _load_reflectance(self, csv_paths, interpolate=False, num_subjects=None):
        """
        Load reflectance data from multiple CSV files.
        If interpolate=True, interpolate the data to match the wavelength range (400–700 nm in 10 nm steps).
        If num_subjects is specified, randomly or sequentially select that many subjects based on `randomize`.
        """
        reflectance_list = []
        for csv_path in csv_paths:
            # Load data from CSV
            data = pd.read_csv(csv_path, skiprows=7 if interpolate else 0, encoding='latin1')
            wavelength = data.iloc[:, 0].to_numpy()  # First column is wavelength
            reflectance = data.iloc[:, 1:].to_numpy()  # Remaining columns are reflectance

            wvl = np.arange(400, 701, 10)  # New wavelength range 
            
            if interpolate:
                if self.verbose:
                    print(f"{'Interpolate reflectance':<30}: {str((bool(interpolate))):>10}")
                # Interpolate reflectance to a new wavelength range (400 to 720 nm, 10 nm step)
                interpolated_reflectance = np.zeros((len(wvl), reflectance.shape[1]))
                for i in range(reflectance.shape[1]):
                    interp_func = interp1d(wavelength, reflectance[:, i], kind='cubic', bounds_error=False, fill_value="extrapolate")
                    interpolated_reflectance[:, i] = interp_func(wvl)
                reflectance_list.append(interpolated_reflectance)
            else:
                # Ensure the wavelength matches the expected range
                if not np.array_equal(wavelength, wvl):
                    raise ValueError(f"Wavelengths in {csv_path} do not match the expected range.")
                reflectance_list.append(reflectance)

        # Combine all reflectance data
        combined_reflectance = np.concatenate(reflectance_list, axis=1)

        # If num_subjects is specified, select that many subjects
        if num_subjects:
            total_subjects = combined_reflectance.shape[1]
            if self.verbose:
                #print(f"{'Available ' + ('real' if self.isRealSkin else 'fake') + ' skin subjects':<30}: {total_subjects:>10}")
                print(f"{'Selected subjects':<30}: {f'{num_subjects} / {total_subjects}':>10}")
                print(f"{'Randomly select':<30}: {str(bool(self.randomize)):>10}")
            if num_subjects > total_subjects:
                raise ValueError(f"num_subjects ({num_subjects}) exceeds the number of available subjects ({total_subjects}).")
            if num_subjects < 1:
                raise ValueError("num_subjects must be at least 1.")
            
            if self.randomize:
                selected_indices = np.random.choice(total_subjects, num_subjects, replace=False)
            else:
                selected_indices = np.arange(num_subjects)

            if self.verbose:
                pass # print(f"{'Selected indices'}: {selected_indices}")
 
            # Select the reflectance data for the chosen subjects
            combined_reflectance = combined_reflectance[:, selected_indices]    
            combined_reflectance = np.vstack((selected_indices, combined_reflectance)) # (32, num_subject) first row is the index

        if self.verbose:
            print(f"Reflectance data loading complete".center(42, '-'))

        return combined_reflectance

    def _apply_illumination(self, reflectance_cube):
        """
        Apply random illumination to the reflectance cube using RandomIllumination.
        """
        # Convert reflectance_cube to a PyTorch tensor and permute to (bands, H, W)
        reflectance_tensor = torch.tensor(reflectance_cube, dtype=torch.float32).permute(2, 0, 1)  # (bands, H, W)
        
        # Apply random illumination
        augmented_tensor, illuminant = self.random_illum(reflectance_tensor)

        # Convert back to numpy and permute to (H, W, bands)
        illuminated_cube = augmented_tensor.permute(1, 2, 0).numpy()  # Convert back to (H, W, bands)

        return illuminated_cube, illuminant

    def _apply_sensor_sensitivity(self, reflectance_cube):
        """
        Apply sensor sensitivity to the reflectance cube.
        """
        # Integrate with sensor sensitivity
        intensity_cube = np.einsum('hwl,cl->hwc', reflectance_cube, self.sensor_sens)
        return intensity_cube

    def _add_sensor_noise(self, intensity_cube):
        """
        Add shot-like sensor noise to the intensity cube.
        """
        alpha = self.noise_scale
        std_dev = alpha * np.sqrt(np.clip(intensity_cube, 1e-10, None))
        noise = np.random.normal(loc=0.0, scale=std_dev)
        intensity_cube_noisy = intensity_cube + noise
        intensity_cube_noisy = np.clip(intensity_cube_noisy, 0.0, 1e3)  # Clip to a valid range
        return intensity_cube_noisy

    def _generate_patch(self):
        """
        Generate a patch for the given data type ('real' or 'fake') and subject ID.
        """
        # Use the loaded reflectance data
        reflectance_data = self.reflectance_data[1:]
        reflectance_idx = self.reflectance_data[0]

        # If subject_id is not provided, randomly select one
        idx = np.random.randint(0, reflectance_data.shape[1])

        # Sample reflectance values with noise
        reflectance_cube = np.random.normal(
            loc=reflectance_data[:, idx],
            scale=self.sr**2,
            size=(self.patch_size, self.patch_size, reflectance_data.shape[0])
        )
        
        # Apply illumnation spectrum
        if self.applyRandomIllumination:
            reflectance_cube, illuminant = self._apply_illumination(reflectance_cube)
            
        # Apply sensor sensitivity
        intensity_cube = self._apply_sensor_sensitivity(reflectance_cube)

        # Add sensor noise
        if self.applyNoise:
            intensity_cube = self._add_sensor_noise(intensity_cube)

        # Convert to PyTorch tensor and permute to (bands, H, W)
        patch_tensor = torch.tensor(intensity_cube, dtype=torch.float32)
        patch_tensor = patch_tensor.permute(2, 0, 1)  # (bands, H, W)

        return patch_tensor, int(reflectance_idx[idx]), illuminant if self.applyRandomIllumination else 'None'
    
    def __getitem__(self, idx):
        """
        Get a patch and its label by index.
        """
        # Generate a patch
        patch_tensor, csv_column_idx, illuminant = self._generate_patch()

        if self.transform:
            patch_tensor = self.transform(patch_tensor)

        # Assign label based on skin type
        label = 0 if self.isRealSkin else 1

        # Return the patch, label, and global index
        if hasattr(self, 'global_offset'):
            global_idx = idx + self.global_offset
        else:
            global_idx = idx

        if self.verbose:
            print(f"Generating Patch".center(32, '-'))
            print(f"{'Index':<20}: {idx:>10}")
            print(f"{'Skin Type':<20}: {'Real' if self.isRealSkin else 'Fake':>10}")
            print(f"{'Patch size':<20}: {f'{self.patch_size} x {self.patch_size}':>10}")
            print(f"{'Reflectance idx':<20}: {csv_column_idx:>10}")
            print(f"{'Illuminant':<20}: {illuminant[:8]:>10}")
            print("-"*32)

        return patch_tensor, label, global_idx

    def __len__(self):
        return self.total_patches