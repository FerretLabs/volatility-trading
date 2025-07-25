import os
import sys
import time
import pickle

import matplotlib.pyplot as plt

import seaborn as sns

import jax
from jax import random
import jax.numpy as jnp

import pandas as pd
import numpy as np

import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.infer.util import Predictive

######################
# See Simon, D.P. and Campasano, J., 2014. The VIX futures basis: Evidence and trading strategies.
######################

num_samples = 10**3
num_chains = 12
numpyro.set_platform("cpu")
numpyro.set_host_device_count(num_chains)
rng_key = jax.random.PRNGKey(0)

def model(x, y):
    alpha_0 = numpyro.sample("alpha_0", dist.Normal(0.0, 0.5))
    alpha_1 = numpyro.sample("alpha_1", dist.Normal(0.0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(.5))
    mu = alpha_0 + alpha_1 * x
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

def run_inference(model, rng_key, x, y):
    start = time.time()
    sampler = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(
        sampler,
        num_warmup=500,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    mcmc.run(rng_key, x=x, y=y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()

df = pd.read_csv(
        "data/vx1_vix_merged_dbe_0.csv",
        header=0,
        index_col=0,
        parse_dates=True,
    ).sort_index(ascending=True)

df = df["2019-01-25":]
print(df["days_until_roll_vx1"].describe())
contango = (df["Close_vix"] / df["Close_vix3m"]).values < 1
basis = (df["Close_vix"] - df["Close_vx1"]).values

for j in range(0, 16):
    alpha_1s = []
    indices = []
    for i in range(j + 1, 18):
        print(f"########{i}########")
        start_of_contract = df["days_until_roll_vx1"] == i
        end_of_contract = df["days_until_roll_vx1"] == j
        num_end_samples = np.sum(end_of_contract)
        num_start_samples = np.sum(start_of_contract)
        if num_start_samples == num_end_samples:
            #vix_end_minus_start  = df["Close_vix"].values[end_of_contract] - df["Close_vix"].values[start_of_contract] 
            #rng_key, rng_key_ = random.split(rng_key)
            #samples = run_inference(model, rng_key_, basis[start_of_contract], vix_end_minus_start)

            vx_end_minus_start = df["Close_vx1"].values[end_of_contract] - df["Close_vx1"].values[start_of_contract]
            rng_key, rng_key_ = random.split(rng_key)
            samples = run_inference(model, rng_key_, basis[start_of_contract], vx_end_minus_start)

            alpha_1s.append(np.array(samples["alpha_1"]))
            indices.append(np.full(num_samples * num_chains, i))

    if len(alpha_1s) > 1:
        distributions = np.concatenate(alpha_1s, axis=0).reshape(-1, 1)
        long_indices = np.concatenate(indices, axis=0).reshape(-1, 1)
        plot_data = np.concatenate([long_indices, distributions], axis=1)
        plot_df = pd.DataFrame(plot_data, columns=["days_until_expiry", "alpha_1"])
        plt.clf()
        # sns.scatterplot(x=[x for x in range(1, 1 + len(alpha_1s))], y=alpha_1s)
        sns.violinplot(x=plot_df["days_until_expiry"], y=plot_df["alpha_1"], native_scale=True, color="blue")
        plt.axhline(y=0, color="red", linestyle="--")
        plt.savefig(f"plots/vx_alpha_1s_eoc_{j}.png")























