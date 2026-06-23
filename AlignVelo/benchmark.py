from typing import List, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

from anndata import AnnData
from scvelo import logging as logg
import pandas as pd

#-------------------------------------------------------------------------------------------
#please run this before computing cross_boundary_correctness
# 使用示例:
# 假设你已经有了adata对象且计算了neighbors
# adata = sc.read("your_data.h5ad")
# scv.pp.neighbors(adata)
# reassign_celltype_by_neighbors(adata, cell_type_key='cell_type')
def reassign_celltype_by_neighbors(adata, cell_type_key='cell_type'):
    """
    根据每个细胞的最近邻细胞类型重新赋值当前细胞的类型
    """ 
    # 获取所有细胞的邻居索引
    try:
        neighbor_indices = adata.uns['neighbors']['indices']
    except Exception as e: 
        raise ValueError("请先运行scv.pp.neighbors()计算邻域关系")
      
    # 获取当前所有细胞的类型
    current_types = adata.obs[cell_type_key].values
    
    # 为每个细胞找到邻居中最常见的类型
    new_types = []
    for i in range(adata.shape[0]):
        # 获取当前细胞的邻居索引（不包括自己）
        neighbors = neighbor_indices[i]
        
        # 获取邻居细胞的类型
        neighbor_types = current_types[neighbors]
        
        # 找到最常见的类型
        if len(neighbor_types) > 0:
            mode_result = stats.mode(neighbor_types)
            most_common = mode_result.mode[0]
            new_types.append(most_common)
        else:
            # 如果没有邻居，保留原类型
            new_types.append(current_types[i])
    
    # 创建新列保存重新分配的类型
    adata.obs[f'reassigned_{cell_type_key}'] = pd.Categorical(new_types)
    
    return adata

def inner_cluster_coh(adata, k_cluster, k_velocity, return_raw=False):
    """In-cluster Coherence Score.
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}
    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns['neighbors']['indices'][sel]
        same_cat_nodes = map(lambda nodes:keep_type(adata, nodes, cat, k_cluster), nbs)
        velocities = adata.layers[k_velocity]
        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean() 
                     for ith, nodes in enumerate(same_cat_nodes) 
                     if len(nodes) > 0]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)
    
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])


def inner_cluster_coh_batch(adata, k_cluster, k_velocity, k_batch, return_raw=False):
    """In-cluster Coherence Score specific for batch consistency.
    
    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        k_batch (str): key to batch in adata.obs DataFrame
        return_raw (bool): return aggregated or raw scores.
        
    Returns:
        dict: all_scores indexed by cluster_edges
        or
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}
    
    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        cat_batches = adata.obs[k_batch].values[sel]
        batches = np.unique(cat_batches)
        
        mean_score=[]
        all_scores[cat]=[]
        for batch in batches:
            sel = (adata.obs[k_cluster] == cat)&(adata.obs[k_batch] == batch)
            nbs = adata.uns['neighbors']['indices'][sel]
            same_cat_nodes = map(lambda nodes:keep_type2(adata, nodes, cat, batch, k_cluster, k_batch), nbs)
        
            velocities = adata.layers[k_velocity]
            cat_vels = velocities[sel]
            cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean() 
                        for ith, nodes in enumerate(same_cat_nodes) 
                        if len(nodes) > 0]
            all_scores[cat].append(cat_score) 
            mean_score.append(np.mean(cat_score))
        scores[cat] = np.mean(mean_score)
    
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])


# Code modified from cross_boundary_correctness
# https://github.com/qiaochen/VeloAE/blob/main/veloproj/eval_util.py
def _select_emb(adata: AnnData, k_velocity: str, x_emb_key: str):
    if x_emb_key in adata.layers.keys():
        # using embedding from raw space
        x_emb = adata.layers[x_emb_key]
        v_emb = adata.layers[k_velocity]

    else:  # embedding from visualization dimensions
        if x_emb_key.startswith("X_"):
            v_emb_key = k_velocity + x_emb_key[1:]
        else:
            v_emb_key = k_velocity + "_" + x_emb_key
            x_emb_key = "X_" + x_emb_key
        assert x_emb_key in adata.obsm.keys()
        assert v_emb_key in adata.obsm.keys()
        x_emb = adata.obsm[x_emb_key]
        v_emb = adata.obsm[v_emb_key]
    return x_emb, v_emb

def cross_boundary_correctness(
    adata: AnnData,
    k_cluster: str,
    k_velocity: str,
    cluster_edges: List[Tuple[str, str]],
    return_raw: bool = False,
    x_emb_key: str = "umap",
    inplace: bool = True,
    output_key_prefix: str = "",
):
    """Cross-Boundary Direction Correctness Score (A->B)

    Args:
        adata (Anndata): Anndata object.
        k_cluster (str): key to the cluster column in adata.obs DataFrame.
        k_velocity (str): key to the velocity matrix in adata.obsm.
        cluster_edges (list of tuples("A", "B")): pairs of clusters has transition direction A->B
        return_raw (bool): return aggregated or raw scores.
        x_emb (str): key to x embedding. If one of the keys in adata.layers, then
            will use the layer and compute the score in the raw space. Otherwise,
            will use the embedding in adata.obsm. Default to "umap".
        inplace (bool): whether to add the score to adata.obs.
        output_key_prefix (str): prefix to the output key. Defaults to "".

    Returns:
        dict: mean scores indexed by cluster_edges
        float: averaged score over all cells.
        and
        dict: all_scores indexed by cluster_edges if return_raw is True.
    """

    scores = {}
    all_scores = {}
    x_emb, v_emb = _select_emb(adata, k_velocity, x_emb_key)

    for u, v in cluster_edges:
        assert u in adata.obs[k_cluster].cat.categories, f"cluster {u} not found"
        assert v in adata.obs[k_cluster].cat.categories, f"cluster {v} not found"
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns["neighbors"]["indices"][sel]  # [n * 30]

        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.mean(dir_scores))

        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score

    all_cbcs_ = np.concatenate([np.array(d) for d in all_scores.values()])

    if inplace:
        adata.uns[f"{output_key_prefix}direction_scores"] = scores
        adata.uns[f"{output_key_prefix}raw_direction_scores"] = all_cbcs_
        logg.info(f"added '{output_key_prefix}direction_scores' (adata.uns)")
        logg.info(f"added '{output_key_prefix}raw_direction_scores' (adata.uns)")

    if return_raw:
        return scores, np.mean(all_cbcs_), all_scores

    return scores, np.mean(all_cbcs_)


def keep_type(adata, nodes, target, k_cluster):
    """Select cells of targeted type

    Args:
        adata (Anndata): Anndata object.
        nodes (list): Indexes for cells
        target (str): Cluster name.
        k_cluster (str): Cluster key in adata.obs dataframe
    Returns:
        list: Selected cells.
    """
    return nodes[adata.obs[k_cluster][nodes].values == target]

def keep_type2(adata, nodes, target, target2, k_cluster, k_batch):
    """Select cells of targeted type and not target batch

    Args:
        adata (Anndata): Anndata object.
        nodes (list): Indexes for cells
        target (str): Cluster name.
        target2 (str):Batch name.
        k_cluster (str): Cluster key in adata.obs dataframe
    Returns:
        list: Selected cells.
    """
    return nodes[(adata.obs[k_cluster][nodes].values == target) & (adata.obs[k_batch][nodes].values != target2)]

