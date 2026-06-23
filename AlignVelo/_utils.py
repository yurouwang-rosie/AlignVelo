import torch
import numpy as np
import pandas as pd

from pathlib import Path

import json
from typing import Dict

from itertools import repeat
from collections import OrderedDict
from collections.abc import Mapping
from scvelo.core import sum as sum_
from anndata import AnnData


##calculate KL divergence
def normal_kl(mu1, lv1, mu2, lv2):
    """
    Calculate KL divergence
    This function is from torchdiffeq: https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
    """
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1/2.
    lstd2 = lv2/2.

    kl = lstd2 - lstd1 + (v1 + (mu1-mu2)**2.)/(2.*v2) - 0.5
    return kl

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    
def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader

def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def update_dict(d: Dict, u: Mapping, copy=False):
    """recursively updates nested dict with values from u."""
    if copy:
        d = d.copy()
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = update_dict(d.get(k, {}), v, copy)
            d[k] = r
        else:
            d[k] = u[k]
    return d

def autoset_coeff_s(spliced_matrix: torch.Tensor, unspliced_matrix: torch.Tensor, use_raw: bool = True) -> float:
    """
    Automatically set the weighting for objective term of the spliced
    read correlation. Modified from the scv.pl.proportions function.

    Args:
        u : unspliced RNA matrix.
        s : spliced RNA matrix.
        use_raw (bool): use raw data or processed data.

    Returns:
        float: weighting coefficient for objective term of the unpliced read

    This function is from autoset_coeff_s: https://github.com/bowang-lab/DeepVelo/blob/main/deepvelo/utils/preprocess.py
    """
    spliced_counts = torch.sum(spliced_matrix, dim=1)
    unspliced_counts = torch.sum(unspliced_matrix, dim=1)
    
    # Total counts for each cell
    counts_total = spliced_counts + unspliced_counts
    counts_total = counts_total + (counts_total == 0).float()
    spliced_ratio = spliced_counts / counts_total
    unspliced_ratio = unspliced_counts / counts_total
    mean_spliced_ratio = torch.mean(spliced_ratio).item()
    mean_unspliced_ratio = torch.mean(unspliced_ratio).item()

    ratio = mean_spliced_ratio / (mean_spliced_ratio + mean_unspliced_ratio)

    if ratio < 0.7:
        coeff_s = 0.5
        print(
            f"The ratio of spliced reads is {ratio*100:.1f}% (less than 70%). "
            f"Suggest using coeff_s {coeff_s}."
        )
    elif ratio < 0.85:
        coeff_s = 0.75
        print(
            f"The ratio of spliced reads is {ratio*100:.1f}% (between 70% and 85%). "
            f"Suggest using coeff_s {coeff_s}."
        )
    else:
        coeff_s = 1.0
        print(
            f"The ratio of spliced reads is {ratio*100:.1f}% (more than 85%). "
            f"Suggest using coeff_s {coeff_s}."
        )

    return coeff_s

from scvelo.preprocessing.neighbors import _get_rep, _set_pca, get_duplicate_cells
from scvelo import logging as logg
import scanpy as sc
from scipy.sparse import csr_matrix, bmat
def neighbor(adata,
             n_knn_neighbors=30,
             batch_key="batch",
             n_pcs=None,
             use_rep=None,
             use_highly_variable=True,
             metric="euclidean"
             ):
    '''
    Find the neighbors within batches.
    '''
    max_n_knn_neighbors = 30
    
    use_rep = _get_rep(adata=adata, use_rep=use_rep, n_pcs=n_pcs)
    if use_rep == "X_pca":
        _set_pca(adata=adata, n_pcs=n_pcs, use_highly_variable=use_highly_variable)

        n_duplicate_cells = len(get_duplicate_cells(adata))
        if n_duplicate_cells > 0:
            logg.warn(
                f"You seem to have {n_duplicate_cells} duplicate cells in your data.",
                "Consider removing these via pp.remove_duplicate_cells.",
            )
    logg.info(f"use_rep : {use_rep}", r=True)

    batch_list = list(adata.obs[batch_key].cat.categories)
    adata_list = [adata[adata.obs[batch_key]==batch].copy() for batch in batch_list]

    m = len(batch_list)
    connectivities_list = [[None for j in range(m)]for i in range(m)] 
    distances_list = [[None for j in range(m)]for i in range(m)]

    # Within batch KNN
    for i in range(len(batch_list)):
        tmp_adata = adata_list[i]
        n_knn_neighbors = min(n_knn_neighbors, max_n_knn_neighbors)
        n_knn_neighbors = max(n_knn_neighbors, 1)
        # actual_max_n_knn_neighbors = max(actual_max_n_knn_neighbors, n_knn_neighbors)
        sc.pp.neighbors(tmp_adata,
                        n_neighbors=n_knn_neighbors,
                        n_pcs=n_pcs,
                        use_rep=use_rep,
                        metric=metric
                        )
        connectivities_list[i][i]=adata_list[i].obsp["connectivities"]
        distances_list[i][i]=adata_list[i].obsp["distances"]

    adata_concat = sc.concat(adata_list)
    adata_concat.obsp["connectivities"] = bmat(connectivities_list).A # 按照位置拼接稀疏矩阵， 后续需要变序号，所以这里转成numpy矩阵
    adata_concat.obsp["distances"] = bmat(distances_list).A
    adata_concat = adata_concat[adata.obs.index]

    adata.uns['neighbors'] = adata_list[0].uns["neighbors"]
    adata.obsp["connectivities"] = csr_matrix(adata_concat.obsp["connectivities"])
    adata.obsp["distances"] = csr_matrix(adata_concat.obsp["distances"])