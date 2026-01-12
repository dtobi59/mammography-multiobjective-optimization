"""
Simple Mammographic Preprocessing Pipeline
Clean, focused implementation for breast cancer detection

Adapted for VinDr-Mammo dataset with stratified selection
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import exposure, filters, morphology, restoration
from skimage.measure import label, regionprops
import pydicom
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class MammogramPreprocessor:
    """
    Simple preprocessing pipeline for mammographic images

    Pipeline stages:
    1. Load DICOM
    2. ROI detection and cropping
    3. Denoising (bilateral filter)
    4. Contrast enhancement (CLAHE)
    5. Resizing to target size
    6. Normalization (standardization)
    """

    def __init__(self, target_size: int = 512, clahe_clip_limit: float = 2.0):
        """
        Initialize preprocessor.

        Args:
            target_size: Output image size (square)
            clahe_clip_limit: Clip limit for CLAHE (higher = more contrast)
        """
        self.target_size = target_size
        self.clahe_clip_limit = clahe_clip_limit

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load DICOM or standard image format.

        Args:
            image_path: Path to image file

        Returns:
            Loaded image as float32 array
        """
        image_path = Path(image_path)

        if image_path.suffix.lower() in ['.dcm', '.dicom']:
            dicom = pydicom.dcmread(image_path)
            img = dicom.pixel_array.astype(np.float32)

            # Handle DICOM inversion (MONOCHROME1)
            if hasattr(dicom, 'PhotometricInterpretation') and \
               dicom.PhotometricInterpretation == 'MONOCHROME1':
                img = np.max(img) - img
        else:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            img = img.astype(np.float32)

        return img

    def detect_roi(self, img: np.ndarray, threshold_percentile: int = 5) -> Tuple[int, int, int, int]:
        """
        Detect breast region of interest (ROI) using Otsu thresholding.

        Args:
            img: Input image
            threshold_percentile: Percentile for initial threshold

        Returns:
            Bounding box (y1, y2, x1, x2)
        """
        # Normalize to 0-255 for processing
        img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

        # Otsu thresholding
        _, binary = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove small objects
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=1000)
        binary = morphology.remove_small_holes(binary, area_threshold=1000)

        # Find largest connected component (breast region)
        labeled = label(binary)
        if labeled.max() == 0:
            # No regions found, return full image
            return 0, img.shape[0], 0, img.shape[1]

        regions = regionprops(labeled)
        largest_region = max(regions, key=lambda r: r.area)

        # Get bounding box with small padding
        y1, x1, y2, x2 = largest_region.bbox

        # Add 5% padding
        h, w = img.shape
        pad_y = int((y2 - y1) * 0.05)
        pad_x = int((x2 - x1) * 0.05)

        y1 = max(0, y1 - pad_y)
        y2 = min(h, y2 + pad_y)
        x1 = max(0, x1 - pad_x)
        x2 = min(w, x2 + pad_x)

        return y1, y2, x1, x2

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Denoise using bilateral filter (preserves edges).

        Args:
            img: Input image

        Returns:
            Denoised image
        """
        # Normalize to 0-255 for bilateral filter
        img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

        # Bilateral filter: smooths while preserving edges
        denoised = cv2.bilateralFilter(img_norm, d=5, sigmaColor=50, sigmaSpace=50)

        return denoised.astype(np.float32)

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            img: Input image

        Returns:
            Contrast-enhanced image
        """
        # Ensure image is in 0-255 range
        img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_norm)

        return enhanced.astype(np.float32)

    def resize(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio with padding.

        Args:
            img: Input image

        Returns:
            Resized image
        """
        h, w = img.shape

        # Calculate scaling factor to fit within target size
        scale = self.target_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create square canvas and center the image
        canvas = np.zeros((self.target_size, self.target_size), dtype=np.float32)

        # Calculate padding
        pad_top = (self.target_size - new_h) // 2
        pad_left = (self.target_size - new_w) // 2

        # Place resized image on canvas
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        return canvas

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image using standardization (zero mean, unit variance).

        Args:
            img: Input image

        Returns:
            Normalized image
        """
        mean = img.mean()
        std = img.std()

        # Avoid division by zero
        if std < 1e-6:
            std = 1.0

        normalized = (img - mean) / std

        return normalized

    def process(self, image_path: str, verbose: bool = False) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Args:
            image_path: Path to image
            verbose: Print progress

        Returns:
            Preprocessed image
        """
        if verbose:
            print(f"\nProcessing: {Path(image_path).name}")

        # Load
        if verbose:
            print(f"Loading...")
        img = self.load_image(image_path)
        if verbose:
            print(f"  Loaded: {img.shape}")

        # Detect ROI
        if verbose:
            print(f"Detecting ROI...")
        y1, y2, x1, x2 = self.detect_roi(img)
        img = img[y1:y2, x1:x2]
        if verbose:
            print(f"  After ROI: {img.shape}")

        # Denoise
        if verbose:
            print(f"Denoising...")
        img = self.denoise(img)

        # Enhance
        if verbose:
            print(f"Enhancing contrast...")
        img = self.enhance_contrast(img)

        # Resize
        if verbose:
            print(f"Resizing to {self.target_size}x{self.target_size}...")
        img = self.resize(img)

        # Normalize
        if verbose:
            print(f"Normalizing...")
        img = self.normalize(img)
        if verbose:
            print(f"  Final: {img.shape}, mean: {img.mean():.6f}, std: {img.std():.6f}")

        return img

    def process_batch(self, image_paths: List[str], verbose: bool = False) -> np.ndarray:
        """
        Process multiple images.

        Args:
            image_paths: List of image paths
            verbose: Print progress

        Returns:
            Stacked array of preprocessed images (N, H, W)
        """
        processed = []
        for i, path in enumerate(image_paths):
            if verbose:
                print(f"\n[{i+1}/{len(image_paths)}]")
            img = self.process(path, verbose=verbose)
            processed.append(img)
        return np.stack(processed)

    def visualize_pipeline(self, image_path: str, save_path: Optional[str] = None):
        """
        Visualize each step of the preprocessing pipeline.

        Args:
            image_path: Path to image
            save_path: Optional path to save visualization
        """
        # Load
        img_original = self.load_image(image_path)

        # ROI
        y1, y2, x1, x2 = self.detect_roi(img_original)
        img_roi = img_original[y1:y2, x1:x2]

        # Denoise
        img_denoised = self.denoise(img_roi)

        # Enhance
        img_enhanced = self.enhance_contrast(img_denoised)

        # Resize
        img_resized = self.resize(img_enhanced)

        # Normalize
        img_normalized = self.normalize(img_resized)

        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        images = [
            (img_original, "1. Original"),
            (img_roi, "2. ROI Detected"),
            (img_denoised, "3. Denoised"),
            (img_enhanced, "4. Contrast Enhanced"),
            (img_resized, "5. Resized"),
            (img_normalized, "6. Normalized")
        ]

        for ax, (img, title) in zip(axes.flatten(), images):
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')

            # Add shape and stats
            stats = f"Shape: {img.shape}\nMean: {img.mean():.2f}\nStd: {img.std():.2f}"
            ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")

        plt.show()


class VinDrMammoDataLoader:
    """
    Data loader for VinDr-Mammo stratified dataset.

    Loads from stratified_selection.csv and preprocesses images.
    """

    def __init__(self,
                 base_dir: str,
                 selection_file: str = 'metadata/stratified_selection.csv',
                 target_size: int = 512,
                 preprocessor: Optional[MammogramPreprocessor] = None):
        """
        Initialize data loader.

        Args:
            base_dir: Base directory (e.g., '/content/drive/MyDrive/vindr-mammo')
            selection_file: Path to stratified selection CSV (relative to base_dir)
            target_size: Target image size
            preprocessor: Custom preprocessor (optional)
        """
        self.base_dir = Path(base_dir)
        self.selection_file = self.base_dir / selection_file

        # Load metadata
        if not self.selection_file.exists():
            raise FileNotFoundError(f"Selection file not found: {self.selection_file}")

        self.df = pd.read_csv(self.selection_file)
        print(f"Loaded {len(self.df)} samples from {self.selection_file}")
        print(f"  Malignant: {(self.df['label'] == 1).sum()}")
        print(f"  Benign:    {(self.df['label'] == 0).sum()}")

        # Preprocessor
        self.preprocessor = preprocessor or MammogramPreprocessor(target_size=target_size)

    def get_image_path(self, idx: int) -> Path:
        """Get full image path for a sample."""
        row = self.df.iloc[idx]
        file_path = row['file_path']
        return self.base_dir / file_path

    def get_label(self, idx: int) -> int:
        """Get label for a sample."""
        return int(self.df.iloc[idx]['label'])

    def get_metadata(self, idx: int) -> dict:
        """Get full metadata for a sample."""
        return self.df.iloc[idx].to_dict()

    def load_sample(self, idx: int, preprocess: bool = True, verbose: bool = False) -> Tuple[np.ndarray, int]:
        """
        Load a single sample.

        Args:
            idx: Sample index
            preprocess: Apply preprocessing
            verbose: Print progress

        Returns:
            (image, label)
        """
        image_path = self.get_image_path(idx)
        label = self.get_label(idx)

        if preprocess:
            image = self.preprocessor.process(str(image_path), verbose=verbose)
        else:
            image = self.preprocessor.load_image(str(image_path))

        return image, label

    def load_batch(self, indices: List[int], preprocess: bool = True, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a batch of samples.

        Args:
            indices: List of sample indices
            preprocess: Apply preprocessing
            verbose: Print progress

        Returns:
            (images, labels) as numpy arrays
        """
        images = []
        labels = []

        for i, idx in enumerate(indices):
            if verbose:
                print(f"\nLoading sample {i+1}/{len(indices)} (index {idx})")

            image, label = self.load_sample(idx, preprocess=preprocess, verbose=verbose)
            images.append(image)
            labels.append(label)

        return np.stack(images), np.array(labels)

    def get_train_val_test_split(self,
                                  train_ratio: float = 0.7,
                                  val_ratio: float = 0.15,
                                  test_ratio: float = 0.15,
                                  random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Patient-level train/val/test split.

        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility

        Returns:
            (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        np.random.seed(random_seed)

        # Get unique patients
        patients = self.df['study_id'].unique()
        np.random.shuffle(patients)

        # Calculate split points
        n_train = int(len(patients) * train_ratio)
        n_val = int(len(patients) * val_ratio)

        # Split patients
        train_patients = patients[:n_train]
        val_patients = patients[n_train:n_train + n_val]
        test_patients = patients[n_train + n_val:]

        # Split dataframe
        train_df = self.df[self.df['study_id'].isin(train_patients)].reset_index(drop=True)
        val_df = self.df[self.df['study_id'].isin(val_patients)].reset_index(drop=True)
        test_df = self.df[self.df['study_id'].isin(test_patients)].reset_index(drop=True)

        print(f"\nPatient-level split:")
        print(f"  Train: {len(train_patients)} patients, {len(train_df)} images")
        print(f"  Val:   {len(val_patients)} patients, {len(val_df)} images")
        print(f"  Test:  {len(test_patients)} patients, {len(test_df)} images")

        print(f"\nLabel distribution:")
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            mal = (df['label'] == 1).sum()
            ben = (df['label'] == 0).sum()
            print(f"  {name}: {mal} malignant ({mal/len(df)*100:.1f}%), {ben} benign ({ben/len(df)*100:.1f}%)")

        return train_df, val_df, test_df

    def __len__(self):
        """Number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get a sample (for compatibility with PyTorch)."""
        return self.load_sample(idx, preprocess=True, verbose=False)


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = MammogramPreprocessor(target_size=512)

    # Example: Process a single image
    # image = preprocessor.process('/path/to/image.dicom', verbose=True)

    # Example: Visualize pipeline
    # preprocessor.visualize_pipeline('/path/to/image.dicom')

    # Example: Load VinDr-Mammo dataset
    # loader = VinDrMammoDataLoader(
    #     base_dir='/content/drive/MyDrive/vindr-mammo',
    #     target_size=512
    # )

    # Get train/val/test split
    # train_df, val_df, test_df = loader.get_train_val_test_split()

    # Load a batch
    # images, labels = loader.load_batch([0, 1, 2, 3, 4])

    print("Mammogram preprocessing pipeline ready!")
