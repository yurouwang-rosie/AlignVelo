from typing import Optional
import torch
from AlignVelo._utils import autoset_coeff_s

def pearson(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    """
    The pearson correlation coefficient between two vectors. by default, the
    first dimension is the sample dimension.

    Args:
    x (torch.Tensor):
        The first vector, (N, D).
    y (torch.Tensor):
        The second vector, (N, D).
    mask (torch.BoolTensor):
        The mask showing valid data positions, (N, D).

    Returns: (torch.Tensor), (D,)
    """
    if mask is None:
        x = x - torch.mean(x, dim=0)
        y = y - torch.mean(y, dim=0)
        x = x / (torch.std(x, dim=0) + 1e-9)
        y = y / (torch.std(y, dim=0) + 1e-9)
        return torch.mean(x * y, dim=0)  # (D,)
    else:
        assert mask.dtype == torch.bool
        mask = mask.detach().float()
        num_valid_data = torch.sum(mask, dim=0)  # (D,)

        y = y * mask
        x = x * mask
        x = x - torch.sum(x, dim=0) / (num_valid_data + 1e-9)
        y = y - torch.sum(y, dim=0) / (num_valid_data + 1e-9)
        y = y * mask  # make the invalid data to zero again to ignore
        x = x * mask
        x = x / torch.sqrt(torch.sum(torch.pow(x, 2), dim=0) + 1e-9)
        y = y / torch.sqrt(torch.sum(torch.pow(y, 2), dim=0) + 1e-9)
        return torch.sum(x * y, dim=0)  # (D,)
    
def direction_loss(
    velocity: torch.Tensor,
    spliced_counts: torch.Tensor,
    unspliced_counts: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """
    The constraint for the direction of the velocity vectors. Large ratio of u/s
    should have positive direction, for each gene.

    Args:
    velocity (torch.Tensor):
        The predicted velocity vectors, (batch_size, genes).
    spliced_counts (torch.Tensor):
        The number of spliced reads, (batch_size, genes).
    unspliced_counts (torch.Tensor):
        The number of unspliced reads, (batch_size, genes).
    velocity_u (torch.Tensor):
        The predicted velocity vectors for unspliced reads, (batch_size, genes).
    reduce (bool):
        Whether to reduce the loss to a scalar.

    Returns: (torch.Tensor), (1,)
    """
    coeff_s = autoset_coeff_s(spliced_counts, unspliced_counts)
    coeff_u = 1.0
    # 3. Intereting for the gliogenic cells of the GABAInterneuraons ====
    mask1 = unspliced_counts > 0
    mask2 = spliced_counts > 0
    corr = coeff_u * pearson(velocity, unspliced_counts, mask1) - coeff_s * pearson(
        velocity, spliced_counts, mask2
    )

    if not reduce:
        if torch.mean(corr / (coeff_u + coeff_s)) >= 0 :
            reverse = False
        else:
            reverse = True
        return torch.mean(corr / (coeff_u + coeff_s)), reverse  # range [-1, 1], shape (genes,)

    loss = coeff_u + coeff_s - torch.mean(corr)  # to maximize the correlation
    loss = loss / (coeff_u + coeff_s)
    # should not use correlation to u, since the alpha coefficient is unknown and
    # it definitely varies along the phase portrait.
    # if velocity_u is not None:
    #     corr_unpliced = pearson(velocity_u, unspliced_counts, mask)
    #     loss_u = 1 + torch.mean(corr_unpliced)  # mininize the corr_unpliced
    #     loss = 0.5 * (loss + loss_u)
        
    return loss

###-----------------------------------------
##funtions for UOT
def unbalanced_ot_z(x1, x2, lambda_=0.2, epsilon=0.1,device='cpu',max_iteration={'outer':100,'inner':100},tol=1e-6):
    '''
    Calculate a unbalanced optimal transport matrix between mini subsets.
    Parameters
    ----------
    reg:
        Entropy regularization parameter in OT
    reg_m:
        Unbalanced OT parameter. Larger values means more balanced OT
    device
        training device

    Returns
    -------
    matrix
        mini-subset unbalanced optimal transport matrix
    '''
    n = x1.size(0)
    m = x2.size(0)
    
    p_n = torch.ones(n, 1)/n
    p_m = torch.ones(m, 1)/m
    
    p_n = p_n.to(device)
    p_m = p_m.to(device)
    pi = p_n @ p_m.T
    cost=clip_distance_matrix(x1, x2, p=2)
    cost_pp = lambda_*geom_loss(x1,x2,pi)+(1-lambda_)*cost
    ot_loss=torch.sum(pi*cost_pp)-epsilon* torch.sum(pi * (torch.log(pi) - 1))
    for j in range(max_iteration["outer"]):
        k = torch.exp(-cost_pp/epsilon)
        g = torch.ones(m,1).to(device)
        f =  torch.ones(n,1).to(device)
        dual_loss=compute_Dualloss(f, k, g, p_n, p_m, epsilon)
        for i in range(max_iteration["inner"]):
            g=1/(n*(k.T@f))
            f=1/(m*(k@g))
            #收敛判断
            dual_loss_new=compute_Dualloss(f, k, g, p_n, p_m, epsilon)
            loss_diff=torch.abs(dual_loss_new-dual_loss)
            if loss_diff<tol:
                break
            dual_loss=dual_loss_new
        pi=torch.diag(f)*k*torch.diag(g)
        cost_pp = lambda_*geom_loss(x1,x2,pi)+(1-lambda_)*cost
        #收敛判断
        ot_loss_new=torch.sum(pi*cost_pp)-epsilon* torch.sum(pi * (torch.log(pi) - 1))
        if torch.abs(ot_loss_new-ot_loss)<tol or not ot_loss_new:
            break
        ot_loss=ot_loss_new
        
    out=pi.detach()
    if torch.isnan(out).sum() > 0:        
        out=None
    return  out

def geom_loss(x1,x2,pi):
    D1=clip_distance_matrix(x1,x1,p=2)
    D2=clip_distance_matrix(x2,x2,p=2)
    row_sums = torch.sum(pi, dim=1).reshape(-1, 1)  # shape [n, 1]
    col_sums = torch.sum(pi, dim=0).reshape(-1, 1)  # shape [m, 1]
    n=D1.shape[0]
    m=D2.shape[0]
    loss=(D1**2@row_sums@torch.ones(1, m, device=pi.device))+torch.ones(n, 1, device=pi.device)@(D2**2@col_sums).T-2*(D1@pi@D2.T)
    return loss

def compute_Dualloss(f, k, g, p_n, p_m, epsilon):
    term1 = -epsilon * torch.sum(torch.diag(f)*k*torch.diag(g))
    term2 = epsilon * torch.sum(torch.log(g)*p_m)+epsilon * torch.sum(torch.log(f)*p_n)
    return term1 + term2

def clip_distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p, while restricting x_i,y_j in unit balls by multiplying a constant.

    Parameters
    ----------
    pts_src
        [R, D] matrix
    pts_dst
        [C, D] matrix
    p
        p-norm
    
    Return
    ------
    [R, C] matrix
         clipped distance matrix
    """
    clip_value=torch.tensor(1000.0)
    max_norm=torch.max(torch.abs(pts_src).max()+  torch.abs(pts_dst).max() , 2*clip_value)/2
    pts_src_clip=clip_value*pts_src/max_norm
    pts_dst_clip=clip_value*pts_dst/max_norm
    x_col = pts_src_clip.unsqueeze(1)
    y_row = pts_dst_clip.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance*0.3