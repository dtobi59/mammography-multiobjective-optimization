"""
Dataset-specific metadata parsers for VinDr-Mammo and INbreast.

VinDr-Mammo and INbreast have different directory structures, metadata formats,
and naming conventions. This module provides parsers to convert each dataset's
native format into a standardized internal format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import xml.etree.ElementTree as ET


def birads_to_binary_label(birads: str) -> int:
    """
    Map BI-RADS category to binary label.

    BI-RADS mapping:
    - 1, 2: Benign (0)
    - 3: Probably benign (0)
    - 4, 4A, 4B, 4C: Suspicious (1)
    - 5: Highly suspicious (1)
    - 6: Biopsy-proven malignancy (1)
    - 0: Incomplete (exclude from dataset)

    Args:
        birads: BI-RADS category as string (e.g., "2", "4A", "4B")

    Returns:
        Binary label: 0 (benign) or 1 (malignant/suspicious)
    """
    # Normalize to uppercase and remove spaces
    birads = str(birads).upper().strip()

    # Remove "BI-RADS" prefix if present (e.g., "BI-RADS 3" -> "3")
    if birads.startswith("BI-RADS"):
        birads = birads.replace("BI-RADS", "").strip()

    # Benign categories
    if birads in ["1", "2", "3"]:
        return 0

    # Suspicious/malignant categories
    if birads in ["4", "4A", "4B", "4C", "5", "6"]:
        return 1

    # Handle edge cases
    if "4" in birads:  # Catch variants like "4-A", "4 A", etc.
        return 1

    if "5" in birads or "6" in birads:
        return 1

    # Default: exclude by raising exception
    raise ValueError(f"Unknown BI-RADS category: {birads}")


class VinDrMammoParser:
    """
    Parser for VinDr-Mammo dataset.

    VinDr-Mammo structure:
    - PNG images converted from DICOM
    - Metadata in CSV format with columns:
      - study_id: Study identifier
      - series_id: Series identifier
      - image_id: Image filename (without extension)
      - laterality: Breast side (L or R)
      - view_position: View type (CC, MLO)
      - breast_birads: BI-RADS assessment
      (actual column names may vary - adjust as needed)
    """

    def __init__(
        self,
        metadata_path: str,
        image_dir: str,
        image_id_col: str = "image_id",
        patient_id_col: str = "study_id",
        laterality_col: str = "laterality",
        view_col: str = "view_position",
        birads_col: str = "breast_birads",
        image_extension: str = ".png",
    ):
        """
        Initialize VinDr-Mammo parser.

        Args:
            metadata_path: Path to VinDr-Mammo CSV metadata file
            image_dir: Directory containing PNG images
            image_id_col: Column name for image ID
            patient_id_col: Column name for patient/study ID
            laterality_col: Column name for breast laterality (L/R)
            view_col: Column name for view type (CC/MLO)
            birads_col: Column name for BI-RADS assessment
            image_extension: Image file extension (default: .png)
        """
        self.metadata_path = metadata_path
        self.image_dir = Path(image_dir)
        self.image_id_col = image_id_col
        self.patient_id_col = patient_id_col
        self.laterality_col = laterality_col
        self.view_col = view_col
        self.birads_col = birads_col
        self.image_extension = image_extension

    def parse(self) -> pd.DataFrame:
        """
        Parse VinDr-Mammo metadata into standardized format.

        Returns:
            DataFrame with standardized columns:
            - image_id: Unique image identifier
            - patient_id: Patient identifier
            - breast_id: Unique breast identifier (patient_id + laterality)
            - view: View type (CC or MLO)
            - label: Binary label (0=benign, 1=malignant)
            - image_path: Relative path to image file
            - birads_original: Original BI-RADS category (for reference)
        """
        # Load metadata
        df = pd.read_csv(self.metadata_path)

        print(f"Loaded VinDr-Mammo metadata: {len(df)} images")

        # Standardize column names
        standardized = pd.DataFrame()

        # Image ID
        standardized["image_id"] = df[self.image_id_col].astype(str)

        # Patient ID
        standardized["patient_id"] = df[self.patient_id_col].astype(str)

        # Laterality (L/R) - normalize to uppercase
        laterality = df[self.laterality_col].astype(str).str.upper().str.strip()

        # Breast ID: Combine patient_id + laterality
        standardized["breast_id"] = (
            standardized["patient_id"] + "_" + laterality
        )

        # View type - normalize to CC/MLO
        view = df[self.view_col].astype(str).str.upper().str.strip()
        standardized["view"] = view.map(lambda x: "CC" if "CC" in x else "MLO")

        # BI-RADS to binary label
        standardized["birads_original"] = df[self.birads_col].astype(str)

        labels = []
        valid_indices = []

        for idx, birads in enumerate(standardized["birads_original"]):
            try:
                label = birads_to_binary_label(birads)
                labels.append(label)
                valid_indices.append(idx)
            except ValueError as e:
                print(f"Warning: Skipping image {standardized['image_id'].iloc[idx]} - {e}")
                continue

        # Filter to valid images only
        standardized = standardized.iloc[valid_indices].reset_index(drop=True)
        standardized["label"] = labels

        # Image path
        standardized["image_path"] = (
            standardized["image_id"] + self.image_extension
        )

        print(f"Valid images after BI-RADS filtering: {len(standardized)}")
        print(f"Label distribution: {standardized['label'].value_counts().to_dict()}")

        # Validate that images exist
        missing_count = 0
        for img_path in standardized["image_path"]:
            full_path = self.image_dir / img_path
            if not full_path.exists():
                missing_count += 1

        if missing_count > 0:
            print(f"Warning: {missing_count} images not found in {self.image_dir}")

        return standardized


class INbreastParser:
    """
    Parser for INbreast dataset.

    INbreast structure:
    - Different directory hierarchy than VinDr-Mammo
    - Metadata may be in CSV or XML format
    - BI-RADS includes subcategories (4A, 4B, 4C)
    - File naming: typically PatientID_view.png or similar
    """

    def __init__(
        self,
        metadata_path: str,
        image_dir: str,
        metadata_format: str = "csv",
        patient_id_col: Optional[str] = "patient_id",
        laterality_col: Optional[str] = "laterality",
        view_col: Optional[str] = "view",
        birads_col: Optional[str] = "birads",
        filename_col: Optional[str] = "file_name",
    ):
        """
        Initialize INbreast parser.

        Args:
            metadata_path: Path to INbreast metadata file (CSV or XML)
            image_dir: Directory containing images
            metadata_format: Format of metadata file ('csv' or 'xml')
            patient_id_col: Column name for patient ID (CSV only)
            laterality_col: Column name for breast laterality (CSV only)
            view_col: Column name for view type (CSV only)
            birads_col: Column name for BI-RADS assessment (CSV only)
            filename_col: Column name for image filename (CSV only)
        """
        self.metadata_path = metadata_path
        self.image_dir = Path(image_dir)
        self.metadata_format = metadata_format.lower()
        self.patient_id_col = patient_id_col
        self.laterality_col = laterality_col
        self.view_col = view_col
        self.birads_col = birads_col
        self.filename_col = filename_col

    def parse(self) -> pd.DataFrame:
        """
        Parse INbreast metadata into standardized format.

        Returns:
            DataFrame with standardized columns (same as VinDrMammoParser)
        """
        if self.metadata_format == "csv":
            return self._parse_csv()
        elif self.metadata_format == "xml":
            return self._parse_xml()
        else:
            raise ValueError(f"Unsupported metadata format: {self.metadata_format}")

    def _parse_csv(self) -> pd.DataFrame:
        """Parse INbreast CSV metadata."""
        df = pd.read_csv(self.metadata_path)

        print(f"Loaded INbreast CSV metadata: {len(df)} images")

        standardized = pd.DataFrame()

        # Image ID - use filename without extension as ID
        if self.filename_col:
            standardized["image_id"] = df[self.filename_col].apply(
                lambda x: Path(x).stem
            )
        else:
            standardized["image_id"] = df.index.astype(str)

        # Patient ID
        standardized["patient_id"] = df[self.patient_id_col].astype(str)

        # Laterality
        if self.laterality_col and self.laterality_col in df.columns:
            laterality = df[self.laterality_col].astype(str).str.upper().str.strip()
        else:
            # Try to infer from filename or other fields
            print("Warning: Laterality not found, attempting to infer from filename")
            laterality = df[self.filename_col].apply(self._infer_laterality)

        # Breast ID
        standardized["breast_id"] = (
            standardized["patient_id"] + "_" + laterality
        )

        # View type
        if self.view_col and self.view_col in df.columns:
            view = df[self.view_col].astype(str).str.upper().str.strip()
            standardized["view"] = view.map(lambda x: "CC" if "CC" in x else "MLO")
        else:
            print("Warning: View not found, attempting to infer from filename")
            standardized["view"] = df[self.filename_col].apply(self._infer_view)

        # BI-RADS to binary label
        standardized["birads_original"] = df[self.birads_col].astype(str)

        labels = []
        valid_indices = []

        for idx, birads in enumerate(standardized["birads_original"]):
            try:
                label = birads_to_binary_label(birads)
                labels.append(label)
                valid_indices.append(idx)
            except ValueError as e:
                print(f"Warning: Skipping image {standardized['image_id'].iloc[idx]} - {e}")
                continue

        # Filter to valid images
        standardized = standardized.iloc[valid_indices].reset_index(drop=True)
        standardized["label"] = labels

        # Image path
        standardized["image_path"] = df[self.filename_col].iloc[valid_indices].values

        print(f"Valid images after BI-RADS filtering: {len(standardized)}")
        print(f"Label distribution: {standardized['label'].value_counts().to_dict()}")

        return standardized

    def _parse_xml(self) -> pd.DataFrame:
        """
        Parse INbreast XML metadata.

        Note: XML structure is dataset-specific. This is a template
        that should be adapted to the actual XML schema.
        """
        tree = ET.parse(self.metadata_path)
        root = tree.getroot()

        records = []

        # Example XML parsing - adjust based on actual schema
        for patient in root.findall(".//patient"):
            patient_id = patient.get("id") or patient.findtext("id")

            for image in patient.findall(".//image"):
                filename = image.findtext("filename")
                laterality = image.findtext("laterality", "").upper()
                view = image.findtext("view", "").upper()
                birads = image.findtext("birads", "")

                records.append({
                    "patient_id": patient_id,
                    "filename": filename,
                    "laterality": laterality,
                    "view": view,
                    "birads": birads,
                })

        df = pd.DataFrame(records)

        # Convert to standardized format using similar logic as CSV
        return self._parse_csv_like(df)

    def _parse_csv_like(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame to standardized format."""
        standardized = pd.DataFrame()

        standardized["image_id"] = df["filename"].apply(lambda x: Path(x).stem)
        standardized["patient_id"] = df["patient_id"].astype(str)

        laterality = df["laterality"].astype(str).str.upper().str.strip()
        standardized["breast_id"] = standardized["patient_id"] + "_" + laterality

        view = df["view"].astype(str).str.upper().str.strip()
        standardized["view"] = view.map(lambda x: "CC" if "CC" in x else "MLO")

        standardized["birads_original"] = df["birads"].astype(str)

        labels = []
        valid_indices = []

        for idx, birads in enumerate(standardized["birads_original"]):
            try:
                label = birads_to_binary_label(birads)
                labels.append(label)
                valid_indices.append(idx)
            except ValueError:
                continue

        standardized = standardized.iloc[valid_indices].reset_index(drop=True)
        standardized["label"] = labels
        standardized["image_path"] = df["filename"].iloc[valid_indices].values

        return standardized

    @staticmethod
    def _infer_laterality(filename: str) -> str:
        """Infer breast laterality from filename."""
        filename_upper = str(filename).upper()
        if "_L_" in filename_upper or "_LEFT_" in filename_upper:
            return "L"
        elif "_R_" in filename_upper or "_RIGHT_" in filename_upper:
            return "R"
        else:
            # Default or raise error
            return "UNKNOWN"

    @staticmethod
    def _infer_view(filename: str) -> str:
        """Infer view type from filename."""
        filename_upper = str(filename).upper()
        if "CC" in filename_upper:
            return "CC"
        elif "MLO" in filename_upper:
            return "MLO"
        else:
            return "UNKNOWN"


def parse_dataset(
    dataset_name: str,
    metadata_path: str,
    image_dir: str,
    **kwargs
) -> pd.DataFrame:
    """
    Parse dataset metadata using appropriate parser.

    Args:
        dataset_name: Dataset name ('vindr' or 'inbreast')
        metadata_path: Path to metadata file
        image_dir: Directory containing images
        **kwargs: Additional parser-specific arguments

    Returns:
        Standardized metadata DataFrame
    """
    dataset_name = dataset_name.lower()

    if dataset_name in ["vindr", "vindr-mammo", "vindr_mammo"]:
        parser = VinDrMammoParser(metadata_path, image_dir, **kwargs)
    elif dataset_name in ["inbreast", "in-breast"]:
        parser = INbreastParser(metadata_path, image_dir, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return parser.parse()
