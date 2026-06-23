from copy import deepcopy
from typing import Callable, Mapping
import numpy as np

import torch
from anndata import AnnData
from scvelo import logging as logg

from AlignVelo.parseconfig import ConfigParser
from AlignVelo.model import Trainer
import AlignVelo.data as module_data
import AlignVelo.module as module_arch



# a hack to make constants, see https://stackoverflow.com/questions/3203286
class MetaConstants(type):
    @property
    def default_configs(cls):
        return deepcopy(cls._default_configs)


class Constants(object, metaclass=MetaConstants):
    _default_configs = {
        "name": "AlignVelo_project",
        "n_gpu": 1,  # whether to use GPU
        "arch": {
            "type": "AlignVelo",
            "args": {
            "n_hidden":256,
            "n_latent": 20,  
            "log_variational": False,
            "pred_unspliced": False,
            "batch_norm": "none",
            "layer_norm": "none",
            "kl_weight_z": 0.1,
            "mse_weight": 1,
            "non_zero_weight": 0.95,
            "lambda_": 0.2, 
            "epsilon": 0.5,
            "ot_iteration": {'outer':10,'inner':5},
            "loss1_scale": 1.,
            "loss2_scale": 1.,
            "dropout_rate": 0.1, 
            "scale1":1,
            "scale2":1,       
            },
        },
        "data_loader": {
            "type": "VeloDataLoader",
            "args": {
                "unspliced_layer": "Mu",
                "spliced_layer": "Ms",
                "batch_size": 1024,
                "shuffle": True,
                "validation_split": 0.1,
                "num_workers": 10,
                "velocity_genes": False,
                "use_scaled_u": False,
            },
        },
        "optimizer": {    #default lr
            "type": "Adam",  
            "args": {"lr": 0.001, "weight_decay": 1e-4, "amsgrad": True, "eps":1e-8},
        },
        "parameter_lr": {"lr_decoder":0.003,"lr_velo":0.01},
        "lr_scheduler": {"type": "StepLR", "args": {"step_size": 1, "gamma": 0.97}},
        "trainer": {
            "loss1_epochs": 10,
            "loss2_epochs": 40,
            "loss3_epochs": 50,
            "save_dir": "saved/",
            "save_period": 1000,
            "verbosity": 1,
            "early_stop": 1000,
            "tensorboard": True,
            "grad_clip": True,
        },
    }


def train(
    adata: AnnData,
    batch_key: str,
    configs: Mapping,
    verbose: bool = False,
    return_kinetic_rates: bool = True,
    callback: Callable = None,
    **kwargs,
):
    n_cells, n_gene = adata.shape
    if configs["data_loader"]["args"]["velocity_genes"]:
        n_gene = int(np.sum(adata.var["velocity_genes"]))
    configs["arch"]["args"]["n_gene"] = n_gene
    configs["arch"]["args"]["loss_select"] = "loss2"
    configs["data_loader"]["args"]["batch_key"] = batch_key
    configs["arch"]["args"]["n_batch"] = adata.obs[batch_key].nunique()

    b = adata.obs[batch_key].astype("category")
    counts = b.value_counts(dropna=True)
    top_label = counts.index[0]
    cats = [top_label] + [c for c in b.cat.categories if c != top_label]
    b = b.cat.reorder_categories(cats, ordered=True)
    adata.obs["batch_num"] = b.cat.codes.astype("int64")
    
    gene_thresholds1_list = []
    gene_thresholds2_list = []
    for b in range(configs["arch"]["args"]["n_batch"]):
        idx = adata.obs["batch_num"] == b
        Ms_batch = adata.layers["Ms"][idx]
        Mu_batch = adata.layers["Mu"][idx]
        thr1 = np.percentile(Ms_batch, 99, axis=0) * 0.1
        thr2 = np.percentile(Mu_batch, 99, axis=0) * 0.1
        gene_thresholds1_list.append(torch.from_numpy(thr1))
        gene_thresholds2_list.append(torch.from_numpy(thr2))

    gene_thresholds1 = torch.stack(gene_thresholds1_list)  # (n_batch, n_genes)
    gene_thresholds2 = torch.stack(gene_thresholds2_list)

    gene_thresholds = torch.cat([gene_thresholds1, gene_thresholds2], dim=1) # (n_batch, n_genes*2)

    config = ConfigParser(configs)
    logger = config.get_logger("train")

    # setup data_loader instances, use adata as the data_source to load inmemory data
    data_loader = config.init_obj("data_loader", module_data, data_source=adata)
    valid_data_loader = data_loader.split_validation()
    
    model = config.init_obj("arch", module_arch, n_input=adata.n_vars*2, gene_thresholds=gene_thresholds)
    logger.info(f"Beginning training of {configs['name']} ...")
    if verbose:
        logger.info(configs)
        logger.info(model)

    velo_params = [
        model.scale1, model.scale2, 
        model.beta_mean, model.gamma_mean,
        *list(model.velo_decoder.parameters())  # 解包生成器
        ]
    velo_param_ids = {id(p) for p in velo_params}
    decoder_params = [
        *list(model.x1_decoder.parameters()),
        ]
    decoder_param_ids = {id(p) for p in decoder_params}
    base_params = [p for p in model.parameters() 
              if p.requires_grad and id(p) not in velo_param_ids and id(p) not in decoder_param_ids]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params=[
        {'params':velo_params, 'lr': config["parameter_lr"]["lr_velo"]},
        {'params':decoder_params, 'lr': config["parameter_lr"]["lr_decoder"]},
        {'params':base_params}] 
    trainable_params2=[
        {'params':velo_params, 'lr': config["parameter_lr"]["lr_velo"]},
        {'params':decoder_params, 'lr': config["parameter_lr"]["lr_decoder"]*0.1},
        {'params':base_params}] 
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)
    optimizer2 = config.init_obj("optimizer", torch.optim, trainable_params2)
    lr_scheduler2 = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer2)

    trainer = Trainer(
        model,
        optimizer,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
        lr_scheduler2=lr_scheduler2
    )

    def callback_wrapper(epoch):
        # evaluate all and return the velocity matrix (cells, features)
        config_copy = configs["data_loader"]["args"].copy()
        config_copy.update(shuffle=False, validation_split=0, training=False, data_source=adata)
        eval_loader = getattr(module_data, configs["data_loader"]["type"])(
            **config_copy
        )
        velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, z, u, s = trainer.eval(
            eval_loader, return_kinetic_rates=return_kinetic_rates
        )

        if callback is not None:
            callback(adata, velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, z , u, s, epoch)
        else:
            logg.warn(
                "Set verbose to True but no callback function provided. A possible "
                "callback function accepts at least two arguments: adata, velo_mat "
            )

    if verbose:
        trainer.train_with_epoch_callback(
            callback=callback_wrapper,
            freq=kwargs.get("freq", 30),
        )
    else:
        trainer.train()

    
    config_copy = configs["data_loader"]["args"].copy()
    config_copy.update(shuffle=False, validation_split=0, training=False, data_source=adata)
    eval_loader = getattr(module_data, configs["data_loader"]["type"])(
            **config_copy
        )
    velo_mat, velo_mat_u, alpha_rates, kinetic_rates, pseudotime, z, u, s = trainer.eval(
        eval_loader, return_kinetic_rates=return_kinetic_rates
    )

    print("velo_mat shape:", velo_mat.shape)
    # add velocity
    if configs["data_loader"]["args"]["velocity_genes"]:
        # the predictions only contain the velocity genes
        velocity_ = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
        idx = adata.var["velocity_genes"].values
        velocity_[:, idx] = velo_mat
        if len(velo_mat_u) > 0:
            velocity_u = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
            velocity_u[:, idx] = velo_mat_u
        correct_Mu[:, idx] = u
        correct_Ms[:, idx] = s
    else:
        velocity_ = velo_mat
        velocity_u = velo_mat_u
        correct_Mu = u
        correct_Ms = s

    assert adata.shape == velocity_.shape
    adata.layers["velocity"] = velocity_  # (cells, genes)
    adata.layers["correct_Mu"] = correct_Mu
    adata.layers["correct_Ms"] = correct_Ms
    adata.obs["pseudotime"] = pseudotime
    adata.obsm['z'] = z
    if len(velo_mat_u) > 0:
        adata.layers["velocity_unspliced"] = velocity_u
        logg.hint(f"added 'velocity_unspliced' (adata.layers)")

    logg.hint(f"added 'velocity' (adata.layers)")
    logg.hint(f"added 'correct_Mu' (adata.layers)")
    logg.hint(f"added 'correct_Ms' (adata.layers)")
    logg.hint(f"added 'pseudotime' (adata.obs)")
    logg.hint(f"added 'z' (adata.obsm)")

    if return_kinetic_rates:
        if configs["arch"]["args"]["pred_unspliced"]:
            if configs["data_loader"]["args"]["velocity_genes"]:
                alpha_ = np.full(adata.shape, np.nan, dtype=velo_mat.dtype)
                alpha_[:, adata.var["velocity_genes"].values] = alpha_rates
            else:
                alpha_= alpha_rates
            adata.layers['pred_alpha'] = alpha_
            logg.hint(f"added 'pred_alpha'(adata.layers)")
        for k, v in kinetic_rates.items():
            if v is not None:
                if configs["data_loader"]["args"]["velocity_genes"]:
                    v_ = np.zeros(adata.shape, dtype=v.dtype)
                    v_[adata.var["velocity_genes"].values] = v
                    v = v_
                adata.var["pred_" + k] = v
                logg.hint(f"added 'pred_{k}' (adata.var)")

    logg.hint(f"model scale1: {trainer.model.scale1}")
    logg.hint(f"model scale2: {trainer.model.scale2}")
    return trainer