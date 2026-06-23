import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from anndata import AnnData
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import  SubsetRandomSampler

import scanpy as sc
import scvelo as scv
from AlignVelo._utils import neighbor

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        collate_fn=default_collate,
    ):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert (
                split < self.n_samples
            ), "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
        


class VeloDataset(Dataset):
    def __init__(
        self,
        data_source,
        unspliced_layer,
        spliced_layer,
        batch_key,
        train=True,
        velocity_genes=False,
        use_scaled_u=False,
    ):
        # check if data_source is a file path or inmemory data
        if isinstance(data_source, str):
            data_source = Path(data_source)
            with open(data_source, "rb") as f:
                adata = pickle.load(f)
        elif isinstance(data_source, AnnData):
            adata = data_source
        else:
            raise ValueError("data_source must be a file path or anndata object")
        
        self.Ux_sz = adata.layers[unspliced_layer]
        self.Sx_sz = adata.layers[spliced_layer]
        if velocity_genes:
            self.Ux_sz = self.Ux_sz[:, adata.var["velocity_genes"]]
            self.Sx_sz = self.Sx_sz[:, adata.var["velocity_genes"]]
        if use_scaled_u:
            scaling = np.std(self.Ux_sz, axis=0) / np.std(self.Sx_sz, axis=0)
            self.Ux_sz = self.Ux_sz / scaling

        self.Ux_sz = torch.tensor(self.Ux_sz, dtype=torch.float32)
        self.Sx_sz = torch.tensor(self.Sx_sz, dtype=torch.float32)
    
        thresholds = torch.quantile(self.Ux_sz, 0.95, dim=1)
        self.mask_u = (self.Ux_sz > thresholds.unsqueeze(1)).int()

        self.batch = torch.tensor(adata.obs["batch_num"], dtype=torch.long)
    
    def large_batch(self, device):
        """
        build the large batch for training
        """
        # check if self._large_batch is already built
        if hasattr(self, "_large_batch"):
            return self._large_batch
        self._large_batch = [
            {
                "Ux_sz": self.Ux_sz.to(device),
                "Sx_sz": self.Sx_sz.to(device),
                "batch": self.batch.to(device),
                "mask_u": self.mask_u.to(device),
            }
        ]
        return self._large_batch
    
    def __len__(self):
        return len(self.Ux_sz)

    def __getitem__(self, i):
        data_dict = {
            "Ux_sz": self.Ux_sz[i],
            "Sx_sz": self.Sx_sz[i],
            "batch": self.batch[i],
            "mask_u": self.mask_u[i],
            "index": i,
        }
        return data_dict
    
class VeloDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(
        self,
        data_source,
        unspliced_layer,
        spliced_layer,
        batch_key,
        batch_size, 
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
        velocity_genes=False,
        use_scaled_u=False,
    ):
        self.data_source = data_source
        self.dataset = VeloDataset(
            data_source,
            unspliced_layer,
            spliced_layer,
            batch_key,
            train=training,
            velocity_genes=velocity_genes,
            use_scaled_u=use_scaled_u,
        )
        self.shuffle = shuffle
        self.is_large_batch = batch_size == len(self.dataset)
            
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


import scvelo as scv
from typing import Optional
from sklearn.preprocessing import MaxAbsScaler

def scale_data(
    adata: AnnData,
    obs: Optional[str] = None,
    layers: Optional[str] = None,
    min_max_scale: bool = True,
    filter_on_r2: bool = False,
) -> AnnData:
    """Preprocess data.

    This function removes poorly detected genes and minmax scales the data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    spliced_layer
        Name of the spliced layer.
    unspliced_layer
        Name of the unspliced layer.
    min_max_scale
        Min-max scale spliced and unspliced
    filter_on_r2
        Filter out genes according to linear regression fit

    Returns
    -------
    Preprocessed adata.
    """
    if min_max_scale:
        if layers is not None:
            for layer in layers:
                scaler = MaxAbsScaler()
                adata.layers[layer] = scaler.fit_transform(adata.layers[layer])
        scaler = MaxAbsScaler()
        adata.X = scaler.fit_transform(adata.X)
        if obs is not None:
            for ob in obs:
                scaler = MaxAbsScaler()
                adata.obs[ob] = scaler.fit_transform(adata.obs[[ob]]).ravel()

    n_genes=len(adata.var.index)
    if filter_on_r2:
        scv.tl.velocity(adata, mode="deterministic")

        adata = adata[
            :, np.logical_and(adata.var.velocity_r2 > 0, adata.var.velocity_gamma > 0)
        ].copy()
        adata = adata[:, adata.var.velocity_genes].copy()
    n_genes2=len(adata.var.index)
    print(f"Filtered out {n_genes-n_genes2} genes according to linear regression fit.")
    if n_genes2 < n_genes * 0.5:
        print(f"WARNING: Filtered out too much genes, The data may not fit the model.")

    return adata

def preprocess_data(adata, batch_key, min_shared_counts=20,n_top_genes=2000,n_neighbors=30,n_pcs=30):
    if 'spliced' not in adata.layers or 'unspliced' not in adata.layers:
        print("Please assign 'spliced' and 'unspliced' in adata.layers first.")
    scv.pp.filter_genes(adata, min_shared_counts=min_shared_counts)
    scv.pp.normalize_per_cell(adata)
    scv.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key,subset=True)
    neighbor(adata, n_knn_neighbors=n_neighbors, batch_key=batch_key, n_pcs=n_pcs)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors) ##30
    adata = scale_data(adata, layers=["Ms","Mu"], filter_on_r2=False)
    return adata
