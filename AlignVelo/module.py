import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal
import logging

from AlignVelo._utils import normal_kl
from AlignVelo.loss import unbalanced_ot_z
logger = logging.getLogger(__name__)

def Velo_Euler_func(v, x, t):
    """
    Compute the spliced and unspliced RNA expression based on the velocity and pseudotime using Euler's method.

    Parameters
    ----------
    v
        A given velocity.
    x
        A given expression data.
    t
        A given time point.

    """
    segment_length = 50
    segments = []
    for segment_start in range(0, len(t), segment_length):
        x0 = x [segment_start]
        current_segment = [x0]
        for i in range(1, segment_length):
            if segment_start + i < len(t):
                dt = t[segment_start + i] - t[segment_start + i - 1]
                new_state = current_segment[i - 1] + v[segment_start + i - 1] * dt
                current_segment.append(new_state)
        segments.append(torch.stack(current_segment, dim=0))
    return torch.cat(segments, dim=0)
        
class Latent_Encoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_latent: int = 10,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        batch_norm: bool = False,
        layer_norm: bool = False,
    ):
        super(Latent_Encoder, self).__init__()
        self.n_latent = n_latent
        self.fc = nn.Sequential()
        self.fc.add_module('L1', nn.Linear(n_input, n_hidden))
        if batch_norm:
            self.fc.add_module('N1', nn.BatchNorm1d(n_hidden))
        if layer_norm:
            self.fc.add_module('N2',nn.LayerNorm(n_hidden, elementwise_affine=False))
        if dropout_rate > 0:
            self.fc.add_module('D1',nn.Dropout(p=dropout_rate))
        self.fc.add_module('A1', nn.ReLU())
        self.fc2 = nn.Linear(n_hidden, n_latent*2)

    def to(self, device):
        """
        Move the model to the specified device.
        """
        super().to(device)
        return self
    
    def forward(self, x: torch.Tensor) -> tuple:
        out = self.fc(x)
        out = self.fc2(out)
        qz_mean, qz_logvar = out[:, :self.n_latent], out[:, self.n_latent:]
        return qz_mean, qz_logvar



class X1_Decoder(nn.Module):
    def __init__(
        self,
        n_output: int,
        n_latent: int = 10,
        n_batch: int = 10,
        n_hidden: int = 128,
        batch_norm: bool = False,
        layer_norm: bool = False,
        dropout_rate: float = 0.1,
    ):
        super(X1_Decoder, self).__init__()
    
        self.fc = nn.Sequential()
        self.fc.add_module('L1', nn.Linear(n_latent + n_batch, n_hidden))
        if batch_norm:
            self.fc.add_module('N1', nn.BatchNorm1d(n_hidden))
        if layer_norm:
            self.fc.add_module('N2',nn.LayerNorm(n_hidden, elementwise_affine=False))
        if dropout_rate > 0:
            self.fc.add_module('D1',nn.Dropout(p=dropout_rate))
        self.fc.add_module('A1', nn.ReLU())
        self.mask_branch_s = nn.Sequential(
            nn.Linear(n_hidden, n_output//2),
            nn.Sigmoid()  # 输出概率
        )
        # value branch: 预测非零元素数值
        self.value_branch_s = nn.Linear(n_hidden, n_output//2)
        self.mask_branch_u = nn.Sequential(
            nn.Linear(n_hidden, n_output//2),
            nn.Sigmoid()  # 输出概率
        )
        # value branch: 预测非零元素数值
        self.value_branch_u = nn.Linear(n_hidden, n_output//2)

    def to(self, device):
        super().to(device)
        return self
    
    def forward(self, batch: torch.Tensor, z: torch.Tensor):
        #batch = self.batch_embed(batch)
        #batch = batch + torch.randn_like(batch) * 0.1
        input = torch.cat([batch, z], dim=1)
        out = self.fc(input)
        '''s = self.fc2(out)
        u = self.fc3(out)'''
        mask_pred_s = self.mask_branch_s(out)
        value_pred_s = self.value_branch_s(out)
        mask_pred_u = self.mask_branch_u(out)
        value_pred_u = self.value_branch_u(out)
        s = mask_pred_s * value_pred_s
        u = mask_pred_u * value_pred_u
        output = torch.cat([s, u], dim=1)
        mask_pred = torch.cat([mask_pred_s, mask_pred_u], dim=1)
        value_pred = torch.cat([value_pred_s, value_pred_u], dim=1)
        return output, mask_pred, value_pred



class Velo_Decoder(nn.Module):
    def __init__(
        self,
        n_gene: int,
        n_latent: int = 20,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super(Velo_Decoder, self).__init__()
        self.n_gene = n_gene
        self.fc = nn.Sequential()
        self.fc.add_module('L1', nn.Linear(n_latent, n_hidden))
        if dropout_rate > 0:
            self.fc.add_module('D1',nn.Dropout(p=dropout_rate))
        self.fc.add_module('A1', nn.ReLU())
        self.fc2 = nn.Linear(n_hidden, n_gene)
        self.fc3 = nn.Linear(n_hidden, 1)
        
    def to(self, device):
        """
        Move the model to the specified device.
        """
        super().to(device)
        return self
    
    def forward(self, z: torch.Tensor):
        output = self.fc(z)
        para = self.fc2(output)
        para = nn.Softplus()(para)

        t = self.fc3(output).sigmoid()
        return para, t


class AlignVelo(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_gene: int,
        n_batch: int,
        gene_thresholds: torch.Tensor,
        loss_select: Literal["loss1", "loss2","loss3"],
        n_hidden: int = 256,
        n_latent: int = 20,
        log_variational: bool = False,
        pred_unspliced: bool = False,
        batch_norm: Literal["encoder", "decoder", "both", "none"] = "none",
        layer_norm: Literal["encoder", "decoder", "both", "none"] = "none",
        gamma_init: Optional[np.ndarray] = None,
        kl_weight_z: float = 0.1,
        mse_weight: float = 1,
        non_zero_weight: float = 0.95,
        lambda_: float = 0.2,
        epsilon: float = 0.5,
        ot_iteration: dict = {'outer':50,'inner':5},
        loss1_scale: float = 1.,
        loss2_scale: float = 1.,
        dropout_rate: float = 0.1,
        scale1: float = 1,
        scale2: float = 1,
    ):
        super().__init__()

        self.n_latent = n_latent
        self.log_variational = log_variational
        self.n_input = n_input
        self.n_gene = n_gene
        self.kl_weight_z = kl_weight_z
        self.mse_weight = mse_weight
        self.non_zero_weight = non_zero_weight
        self.lambda_=lambda_
        self.epsilon=epsilon
        self.ot_iteration = ot_iteration
        self.loss1_scale = loss1_scale
        self.loss2_scale = loss2_scale
        self.pred_unspliced = pred_unspliced
        self.loss_select = loss_select
        self.n_batch = n_batch
        self.gene_thresholds = gene_thresholds

        # degradation
        if gamma_init is None:
            self.gamma_mean = torch.nn.Parameter(0.1 * torch.randn(n_gene))
        else:
            self.gamma_mean = torch.nn.Parameter(
                torch.from_numpy(gamma_init)
            )

        # splicing
        # first samples around 1
        self.beta_mean = torch.nn.Parameter(0.1 * torch.randn(n_gene))

        self.current_kinetic_rates = {}
     
        # likelihood dispersion
        # for now, with normal dist, this is just the variance
        
        self.scale1 = torch.nn.Parameter(torch.tensor(scale1, dtype=torch.float32))
        self.scale2 = torch.nn.Parameter(torch.tensor(scale2, dtype=torch.float32))

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        use_batch_norm_encoder = batch_norm == "encoder" or batch_norm == "both"
        use_batch_norm_decoder = batch_norm == "decoder" or batch_norm == "both"
        use_layer_norm_encoder = layer_norm == "encoder" or layer_norm == "both"
        use_layer_norm_decoder = layer_norm == "decoder" or layer_norm == "both"

        self.z_encoder = Latent_Encoder(
            n_input=n_input,
            n_latent=n_latent,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            batch_norm=use_batch_norm_encoder,
            layer_norm=use_layer_norm_encoder
            )
        self.x1_decoder = X1_Decoder(
            n_output=n_input,
            n_latent=n_latent,
            n_batch=n_batch,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            batch_norm=use_batch_norm_decoder,
            layer_norm=use_layer_norm_decoder,
        )
        self.velo_decoder = Velo_Decoder(
            n_gene=n_gene,
            n_latent=n_latent,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        
    def to(self, device):
        """
        Move the model to the specified device.
        """
        super().to(device)

        self.device=device
        self.z_encoder = self.z_encoder.to(device)
        self.x1_decoder = self.x1_decoder.to(device)
        self.velo_decoder = self.velo_decoder.to(device)

        self.beta_mean = self.beta_mean.to(device)
        self.gamma_mean = self.gamma_mean.to(device)
        self.scale1 = self.scale1.to(device)
        self.scale2 = self.scale2.to(device)
        return self
    
    def forward(self, inputdata, batch, mask_u):
        inference_output = self.inference(inputdata, batch)
        generative_output = self.generative(inference_output)
        loss = self.loss(inference_output, generative_output, mask_u)
        velo = inference_output["velo_pre"]
        
        return loss, inference_output["t"], velo

    def inference(
        self,
        inputdata,
        batch,
    ):
        if self.log_variational:
            inputdata_ = torch.log(0.01 + inputdata)
        else:
            inputdata_ = inputdata
        batch_onehot = F.one_hot(batch,num_classes=self.n_batch).to(torch.float32)
    
        qz_mean, qz_logvar = self.z_encoder(inputdata_)
        pz_mean = torch.zeros_like(qz_mean)
        pz_logvar = torch.zeros_like(qz_mean)
        kl_div_z = normal_kl(qz_mean, qz_logvar, pz_mean, pz_logvar).sum(-1).mean()
        epsilon1 = torch.randn(qz_mean.size()).to(self.device)
        z = epsilon1 * torch.exp(.5 * qz_logvar) + qz_mean

        z_dict_new={}
        z_dict={}
        mse_loss = torch.tensor(0.0).to(self.device)
        if self.loss_select in ["loss2"]:
            for j in range(self.n_batch):
                loc_query = torch.where(batch==j)[0]                       
                z_dict[j]=z[loc_query]
                if z_dict[j].size(0)==0:
                    continue
                if j==0 and z_dict[j].size(0)<=10:
                    break
                if j>0:
                    if z_dict[j].size(0)<=10:
                        logger.info('Too few samples')
                        continue
                    pi=unbalanced_ot_z(z_dict[j], z_dict[0], self.lambda_, self.epsilon,device=self.device,max_iteration=self.ot_iteration)
                    if pi==None:
                        logger.info('None OT')
                        continue
                    pi_row_sum = pi.sum(dim=1, keepdim=True)       # 行和 shape (n,1)
                    pi_normalized = pi / (pi_row_sum + 1e-9)
                    z_dict_new[j]=(pi_normalized @z_dict[0]).detach()
                    mse_loss += F.mse_loss(z_dict_new[j], z_dict[j], reduction='none').sum(-1).mean()
         
        rec_x1, mask_pred, value_pred = self.x1_decoder(batch_onehot,z) #非零概率
        # mask loss (BCE)
        mask_target = (inputdata_ > self.gene_thresholds[batch]).float()
        loss_mask = F.binary_cross_entropy(mask_pred, mask_target, reduction='none').sum(-1).mean()
        non_zero_mask = mask_target
        loss_value = F.mse_loss(rec_x1 * non_zero_mask, inputdata_ * non_zero_mask, reduction='none').sum(-1).mean()
        reconst_loss1 = self.non_zero_weight * loss_value + (1 - self.non_zero_weight) * loss_mask
        
        alpha, t = self.velo_decoder(z)
    
        lossz = reconst_loss1 + self.kl_weight_z * kl_div_z + self.mse_weight * mse_loss

        gamma, beta = self._get_rates()
        batch_ref = torch.zeros_like(batch)
        batch_onehot_ref = F.one_hot(batch_ref,num_classes=self.n_batch).to(torch.float32)
        x_pre, _, _ = self.x1_decoder(batch_onehot_ref, z)
        s = x_pre[:,:self.n_gene]
        u = x_pre[:,self.n_gene:]
        ##velo inference
        if self.pred_unspliced:
            pred_s = (beta * u - gamma * s) * self.scale1
            pred_u = (alpha - beta * u) * self.scale2
            mean_pred = torch.cat([pred_s, pred_u], dim=1)
        else:
            mean_pred = (beta * u - gamma * s) * self.scale1

        outputs = {
            "t": t,
            "z": z,
            "x_pre": x_pre,
            "velo_pre": mean_pred,
            "alpha": alpha,
            "gamma": gamma,
            "beta": beta,
            "lossz": lossz
        }
        return outputs

    def _get_rates(self):
        # globals
        # degradation
        gamma = torch.clamp(F.softplus(self.gamma_mean), 0, 50)
        # splicing
        beta = torch.clamp(F.softplus(self.beta_mean), 0, 50)
        self.current_kinetic_rates["beta"] = beta
        self.current_kinetic_rates["gamma"] = gamma

        return gamma, beta

    def generative(self, inference_output):

        t = inference_output["t"].squeeze()
        x = inference_output["x_pre"]
        velo = inference_output["velo_pre"]
        #z = inference_output["z"]
        
        sort_t, sort_ridx = torch.unique(t, return_inverse=True)
        index = torch.argsort(t)
        t = t[index]
        velo = velo[index]
        x = x[index]
        index2 = (t[:-1] != t[1:])
        index2 = torch.cat((index2, torch.tensor([True]).to(self.device))) ## index2 is used to get unique time points as odeint requires strictly increasing/decreasing time points
        t = t[index2]
        x = x[index2]
        t = t.view(-1)

        if not self.pred_unspliced:
            x=x[:,:self.n_gene]
        
        x_pre2 = Velo_Euler_func(velo, x, t)

        reconst_loss = F.mse_loss(x, x_pre2, reduction='none').sum(-1).mean()

        t = t[sort_ridx]
        self.pseudotime = t
        x_pre2 = x_pre2[sort_ridx]
        
        outputs={"x_pre2": x_pre2,
                "reconst_loss": reconst_loss,}
        return outputs

    def loss(
        self,
        inference_outputs,
        generative_outputs,
        mask_u,
    ):
        velo=inference_outputs["velo_pre"]
        unspliced_penalty = 0
        if self.pred_unspliced:
            unspliced_penalty += ((velo[:,self.n_gene:self.n_gene*2]*mask_u)** 2).sum(-1).mean()
        loss1 = self.loss1_scale * inference_outputs["lossz"]
        loss2 = self.loss1_scale * inference_outputs["lossz"]+ self.loss2_scale * (generative_outputs["reconst_loss"] + unspliced_penalty)
        if self.loss_select in ["loss1","loss2"]:
            return loss1
        elif self.loss_select=="loss3":
            return loss2
    
    def sample(
        self,
    ) -> np.ndarray:
        """Not implemented."""
        raise NotImplementedError
        
    def get_kinetic_rates(self):
        return self.current_kinetic_rates