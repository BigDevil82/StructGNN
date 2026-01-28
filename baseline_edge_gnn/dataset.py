"""
Dataset for Edge-based GNN (JSON Format)

Loads pre-processed JSON data and builds PyG Data objects.
"""

import os
from typing import Dict, List, Optional

import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from baseline_edge_gnn.config import data_config
from baseline_edge_gnn.graph_builder import (
    build_graph_from_json,
    get_file_category,
    load_json_data,
)


class EdgeShearWallDataset(InMemoryDataset):
    """
    Edge-based shear wall prediction dataset.

    Loads from a single JSON file containing all samples.
    """

    def __init__(
        self,
        root: str,
        json_path: str,
        file_keys: List[str] = None,
        transform=None,
        pre_transform=None,
        is_test: bool = False,
    ):
        """
        Args:
            root: Cache directory path
            json_path: Path to JSON file containing all samples
            file_keys: List of file keys to include (if None, use all keys in JSON)
            transform: PyG data transform function
            pre_transform: PyG data pre-transform function
            is_test: If True, skip augmentation
        """
        self.json_path = json_path
        self._file_keys_input = file_keys  # Store input file keys
        self.is_test = is_test
        super(EdgeShearWallDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # Load metadata
        metadata_path = self.processed_paths[0].replace(".pt", "_metadata.pt")
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
            self.file_indices = metadata["file_indices"]
            self.aug_modes = metadata["aug_modes"]
            self.file_keys = metadata["file_keys"]
        else:
            self.file_indices = None
            self.aug_modes = None
            self.file_keys = None

    @property
    def raw_file_names(self) -> List[str]:
        """Return raw file names (the JSON file)."""
        return [os.path.basename(self.json_path)]

    @property
    def processed_file_names(self) -> List[str]:
        """Return processed cache file names."""
        return ["edge_data.pt"]

    def process(self):
        """
        Process JSON data into PyG Data objects.
        """
        # Load JSON
        print(f"Loading JSON from: {self.json_path}")
        json_data = load_json_data(self.json_path)

        # Use provided file keys or all keys from JSON
        if self._file_keys_input is not None:
            file_keys = self._file_keys_input
            print(f"Using {len(file_keys)} provided file keys")
        else:
            file_keys = list(json_data.keys())
            print(f"Found {len(file_keys)} samples in JSON")

        data_list = []
        file_indices_list = []
        aug_modes_list = []

        augmentations = data_config.AUGMENTATIONS if not self.is_test else ["none"]

        for file_idx, file_key in enumerate(tqdm(file_keys, desc="Building graphs")):
            for mode in augmentations:
                try:
                    builder = build_graph_from_json(json_data, file_key, mode=mode)

                    if builder is None:
                        continue

                    data = builder.to_pyg_data()

                    if data is None or data.num_nodes == 0:
                        continue

                    # Store metadata in data object
                    data.file_key = file_key
                    data.aug_mode = mode

                    data_list.append(data)
                    file_indices_list.append(file_idx)
                    aug_modes_list.append(mode)

                except Exception as e:
                    print(f"Error processing {file_key} with mode {mode}: {e}")
                    continue

        if len(data_list) == 0:
            raise RuntimeError("No valid graphs were created from JSON data!")

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # Save metadata
        metadata = {
            "file_indices": file_indices_list,
            "aug_modes": aug_modes_list,
            "file_keys": file_keys,
        }
        metadata_path = self.processed_paths[0].replace(".pt", "_metadata.pt")
        torch.save(metadata, metadata_path)

        print(f"Saved {len(data_list)} samples from {len(file_keys)} files")

    def get_file_info(self, idx: int) -> Dict:
        """Get file information for a sample."""
        if self.file_indices is None:
            return {"file": "unknown", "mode": "unknown"}

        return {
            "file": self.file_keys[self.file_indices[idx]],
            "mode": self.aug_modes[idx],
        }

    def get_samples_by_file(self, file_key: str) -> List[int]:
        """Get all sample indices for a specific file (all augmentations)."""
        if self.file_keys is None:
            return []

        try:
            file_idx = self.file_keys.index(file_key)
        except ValueError:
            return []

        return [i for i, f_idx in enumerate(self.file_indices) if f_idx == file_idx]

    def get_sample_weights(self) -> List[float]:
        """
        Get sample weights for WeightedRandomSampler based on file category.

        Returns inverse frequency weights to balance class distribution.
        """
        if self.file_keys is None or self.file_indices is None:
            return [1.0] * len(self)

        # Get category for each sample
        categories = []
        for file_idx in self.file_indices:
            file_key = self.file_keys[file_idx]
            cat = get_file_category(file_key)
            categories.append(cat)

        # Count samples per category
        from collections import Counter
        cat_counts = Counter(categories)
        print(f"Category distribution: {dict(cat_counts)}")

        # Compute inverse frequency weights
        total = len(categories)
        num_classes = len(cat_counts)
        weights = []
        for cat in categories:
            if cat == -1:
                weights.append(1.0)
            else:
                # Inverse frequency: fewer samples -> higher weight
                weight = total / (num_classes * cat_counts[cat])
                weights.append(weight)

        return weights
