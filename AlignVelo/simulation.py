import numpy as np
from anndata import AnnData
import random
import pandas as pd

def unspliced(s, s_, gamma, beta):
    """TODO."""
    u=(gamma * s + s_)/beta
    return u

# TODO: Add docstrings
def spliced(tau, a, h, t_):
    """TODO."""
    s = h * np.exp(-a * (tau-t_)**2)
    s_ = -2 * a * (tau-t_) * h * np.exp(-a * (tau-t_)**2)
    return s, s_

def simulation(
    n_obs=300,
    n_vars=None,
    noise_model="normal",
    noise_level=1,
    n_batch=3,
    random_seed=0,
):
    """Simulation of mRNA  kinetics.

    Simulated mRNA metabolism with radial basis function.
    The parameters for each reaction are randomly sampled from a log-normal distribution or uniform distribution,
    and time events follow the Poisson law.

    Returns
    -------
    Returns 'adata'(highly variable genes) and 'adata2'(housekeeping genes) object
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    def draw_poisson(n):
        from random import seed, uniform  # draw from poisson

        seed(random_seed)
        t = np.cumsum([-0.1 * np.log(uniform(0, 1)) for _ in range(n - 1)])
        return np.insert(t, 0, 0)  # prepend t0=0

    def simulate_dynamics(tau, a, h, t_, beta, gamma, noise_model, noise_level, batch, n_batch):
        st, st_ = spliced(tau, a, h, t_)
        ut = unspliced(st, st_, gamma, beta)
        ut, st = np.clip(ut, 0, None), np.clip(st, 0, None)
        true_s = st.copy()
        true_u = ut.copy()
        batch_means = np.random.uniform(-0.5, 0.5, size=n_batch)       # 各批次噪声均值
        batch_vars = np.random.uniform(0.5, 1.0, size=n_batch)         # 各批次噪声缩放因子
        batch_means2 = np.random.uniform(-0.5, 0.5, size=n_batch)      # 各批次噪声均值
        batch_vars2 = np.random.uniform(0.5, 1.0, size=n_batch)        # 各批次噪声缩放因子
        if noise_model == "normal":  # add batch effect and noise
            ut += np.random.normal(
                loc=batch_means[batch], scale=noise_level * batch_vars[batch] * np.percentile(ut, 99) / 10, size=len(ut)
            )
            st += np.random.normal(
                loc=batch_means2[batch], scale=noise_level * batch_vars2[batch] * np.percentile(st, 99) / 10, size=len(st)
            )
        elif noise_model == "uniform":
            ut += batch_means[batch]
            st += batch_means2[batch]   
        ut, st = np.clip(ut, 0, None), np.clip(st, 0, None)
        return ut, st, true_u, true_s, st_

    t = draw_poisson(n_obs)
    t = t/np.max(t)

    # switching time point obtained as fraction of t_max rounded down
    t_ = np.random.uniform(-1, 2, n_vars)
    a = np.random.lognormal(mean=-2, sigma=0.5, size=n_vars)
    h = np.random.lognormal(mean=2, sigma=1, size=n_vars)
    beta = np.random.lognormal(mean=0, sigma=0.3, size=n_vars)
    gamma = np.random.lognormal(mean=0, sigma=0.3, size=n_vars)

    U = np.zeros(shape=(len(t), n_vars))
    S = np.zeros(shape=(len(t), n_vars))
    true_S = np.zeros(shape=(len(t), n_vars))
    true_U = np.zeros(shape=(len(t), n_vars))
    S_ = np.zeros(shape=(len(t), n_vars))

    batch = np.random.randint(0, n_batch, size=n_obs)

    for i in range(n_vars):        
        beta_i = beta[i]
        gamma_i = gamma[i]
        t_i = t_[i]
        a_i = a[i]
        h_i = h[i]
        U[:, i], S[:, i], true_U[:, i], true_S[:, i], S_[:, i] = simulate_dynamics(
            t, a_i, h_i, t_i, beta_i, gamma_i, noise_model, noise_level, batch, n_batch
            )

    batch =  batch.astype(str)

    obs = pd.DataFrame(
        index=[f"cell_{i}" for i in range(n_obs)],
        data={"true_t": t.round(6),
           "batch": batch}
        )
    var = pd.DataFrame(
        index=[f"gene_{i}" for i in range(n_vars)],
        data={
        "true_t_": t_[:n_vars],
        "true_beta": np.ones(n_vars) * beta,
        "true_gamma": np.ones(n_vars) * gamma,
        #"true_scaling": np.ones(n_vars),
        }
    )
    layers = {"unspliced": U, "spliced": S, "true_unspliced": true_U, "true_spliced": true_S, "true_velocity": S_}

    return AnnData(S, obs, var, layers=layers)