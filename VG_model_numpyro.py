#!/usr/bin/env python
# coding: utf-8

# # Variance Gamma Model for VX Futures Log Returns (NumPyro)
# This notebook loads VX futures price data, computes log returns, and fits a Variance Gamma model using NumPyro.

# In[68]: Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jax.scipy.special import logsumexp

def parse_datetime(date, time):
    return pd.to_datetime(date + ' ' + time)

vx_data = pd.read_csv(
    'data/VXMQ25_FUT_CFE.dly_BarData.txt',
    skipinitialspace=True
)
vx_data['DateTime'] = vx_data.apply(lambda row: parse_datetime(row['Date'], row['Time']), axis=1)
vx_data = vx_data.sort_values('DateTime')

# Find the first 17th day of a month in the dataset
first_date = vx_data['DateTime'].min()
first_chunk_start = first_date.replace(day=17)
if first_date.day > 17:
    # If we're past the 17th, move to next month's 17th
    if first_date.month == 12:
        first_chunk_start = first_date.replace(year=first_date.year + 1, month=1, day=17)
    else:
        first_chunk_start = first_date.replace(month=first_date.month + 1, day=17)

# Find the last 9th day of a month in the dataset
last_date = vx_data['DateTime'].max()
last_chunk_end = last_date.replace(day=9)
if last_date.day < 9:
    # If we're before the 9th, move to previous month's 9th
    if last_date.month == 1:
        last_chunk_end = last_date.replace(year=last_date.year - 1, month=12, day=9)
    else:
        last_chunk_end = last_date.replace(month=last_date.month - 1, day=9)

# Filter data to only include complete chunks
original_count = len(vx_data)
vx_data = vx_data[(vx_data['DateTime'] >= first_chunk_start) & (vx_data['DateTime'] <= last_chunk_end)]
dropped_count = original_count - len(vx_data)

print(f"First date in dataset: {first_date.strftime('%Y-%m-%d')}")
print(f"Last date in dataset: {last_date.strftime('%Y-%m-%d')}")
print(f"First chunk starts: {first_chunk_start.strftime('%Y-%m-%d')}")
print(f"Last chunk ends: {last_chunk_end.strftime('%Y-%m-%d')}")
print(f"Dropped {dropped_count} rows (incomplete chunks)")
print(f"Filtered VX data: {len(vx_data)} records")

# Create log returns
log_price = vx_data['Last'].apply(np.log)
log_returns = log_price.diff().dropna().values

# Create chunks from 17th to 9th of next month
def create_monthly_chunks(data, dates):
    """Create chunks from 17th of month to 9th of next month"""
    chunks = []
    chunk_dates = []
    
    current_date = first_chunk_start
    while current_date <= last_chunk_end:
        # Define chunk boundaries
        chunk_start = current_date
        if current_date.month == 12:
            chunk_end = current_date.replace(year=current_date.year + 1, month=1, day=9)
        else:
            chunk_end = current_date.replace(month=current_date.month + 1, day=9)
        
        # Get data for this chunk
        chunk_mask = (dates >= chunk_start) & (dates <= chunk_end)
        chunk_data = data[chunk_mask]
        
        if len(chunk_data) > 0:
            chunks.append(chunk_data)
            chunk_dates.append((chunk_start, chunk_end))
        
        # Move to next month's 17th
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1, day=17)
        else:
            current_date = current_date.replace(month=current_date.month + 1, day=17)
    
    return chunks, chunk_dates

# Create chunks
chunks, chunk_dates = create_monthly_chunks(log_returns, vx_data['DateTime'].iloc[1:])  # Skip first date since we took diff

# Find the maximum chunk size for padding
max_chunk_size = max(len(chunk) for chunk in chunks)
print(f"Number of chunks: {len(chunks)}")
print(f"Maximum chunk size: {max_chunk_size}")

# Pad chunks to same size with NaN
padded_chunks = []
for i, chunk in enumerate(chunks):
    chunk_size = len(chunk)
    if chunk_size < max_chunk_size:
        # Pad with NaN
        padding_size = max_chunk_size - chunk_size
        padded_chunk = np.concatenate([chunk, np.full(padding_size, np.nan)])
    else:
        padded_chunk = chunk
    
    padded_chunks.append(padded_chunk)
    print(f"Chunk {i+1}: {chunk_dates[i][0].strftime('%Y-%m-%d')} to {chunk_dates[i][1].strftime('%Y-%m-%d')} - {chunk_size} days (padded to {max_chunk_size})")

# Convert to numpy array
log_returns_chunked = np.array(padded_chunks)
print(f"Final shape: {log_returns_chunked.shape}")

# For backward compatibility, keep the original log_returns as the flattened version
log_returns = log_returns_chunked.flatten()
plt.figure(figsize=(10, 5))
sns.histplot(np.asarray(log_returns), bins=50, kde=True)
plt.title('Log Returns Distribution')
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.show()

# In[2]
plt.figure(figsize=(10, 5))
plt.plot(vx_data["Last"])
plt.show()

# In[2]
print(np.std(log_returns))
print(np.mean(log_returns))


# In[3]:

# ## Market Microstructure Analysis - Volume by Hour of Day
hourly_vx_data = pd.read_csv(
    "data/VXMQ25_FUT_CFE [C][M]  60 Min  #1_GraphData.txt",
    skipinitialspace=True
)
hourly_vx_data['DateTime'] = hourly_vx_data.apply(lambda row: parse_datetime(row['Date'], row['Time']), axis=1)
hourly_vx_data = hourly_vx_data.sort_values('DateTime')

# Filter hourly data to be greater than April 2nd, 2025
filter_date = pd.to_datetime('2025-04-02')
hourly_vx_data = hourly_vx_data[hourly_vx_data['DateTime'] > filter_date]
print(f"Filtered hourly VX data: {len(hourly_vx_data)} records after {filter_date.strftime('%Y-%m-%d')}")

# Extract hour from DateTime for volume analysis
hourly_vx_data['Hour'] = hourly_vx_data['DateTime'].dt.hour

# Calculate volume statistics by hour
hourly_volume = hourly_vx_data.groupby('Hour')['Volume'].agg(['mean', 'std', 'count', 'sum']).reset_index()
hourly_volume.columns = ['Hour', 'Mean_Volume', 'Std_Volume', 'Trade_Count', 'Total_Volume']

# Plot volume patterns by hour
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Mean volume by hour
axes[0, 0].bar(hourly_volume['Hour'], hourly_volume['Mean_Volume'])
axes[0, 0].set_title('Average Volume by Hour of Day')
axes[0, 0].set_xlabel('Hour')
axes[0, 0].set_ylabel('Mean Volume')
axes[0, 0].grid(True, alpha=0.3)

# Total volume by hour
axes[0, 1].bar(hourly_volume['Hour'], hourly_volume['Total_Volume'])
axes[0, 1].set_title('Total Volume by Hour of Day')
axes[0, 1].set_xlabel('Hour')
axes[0, 1].set_ylabel('Total Volume')
axes[0, 1].grid(True, alpha=0.3)

# Number of trades by hour
axes[1, 0].bar(hourly_volume['Hour'], hourly_volume['Trade_Count'])
axes[1, 0].set_title('Number of Trades by Hour of Day')
axes[1, 0].set_xlabel('Hour')
axes[1, 0].set_ylabel('Trade Count')
axes[1, 0].grid(True, alpha=0.3)

# Volume volatility by hour
axes[1, 1].bar(hourly_volume['Hour'], hourly_volume['Std_Volume'])
axes[1, 1].set_title('Volume Volatility by Hour of Day')
axes[1, 1].set_xlabel('Hour')
axes[1, 1].set_ylabel('Volume Standard Deviation')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("Market Microstructure Analysis:")
print(f"Total trading hours: {len(hourly_volume)}")
print(f"Peak volume hour: {hourly_volume.loc[hourly_volume['Mean_Volume'].idxmax(), 'Hour']:.0f}:00")
print(f"Lowest volume hour: {hourly_volume.loc[hourly_volume['Mean_Volume'].idxmin(), 'Hour']:.0f}:00")

# Calculate intraday volume profile
print("\nIntraday Volume Profile:")
for _, row in hourly_volume.iterrows():
    print(f"Hour {row['Hour']:02.0f}:00 - Mean Volume: {row['Mean_Volume']:8.0f}, "
          f"Total Volume: {row['Total_Volume']:10.0f}, "
          f"Trades: {row['Trade_Count']:4.0f}")

# Additional market microstructure analysis
print("\n" + "="*60)
print("ADDITIONAL MARKET MICROSTRUCTURE ANALYSIS")
print("="*60)

# Analyze returns by hour of day
hourly_vx_data['Log_Return'] = hourly_vx_data['Last'].apply(np.log).diff()
hourly_returns = hourly_vx_data.groupby('Hour')['Log_Return'].agg(['mean', 'std', 'count']).reset_index()
hourly_returns.columns = ['Hour', 'Mean_Return', 'Return_Std', 'Return_Count']

# Plot returns by hour
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Mean returns by hour
axes[0].bar(hourly_returns['Hour'], hourly_returns['Mean_Return'])
axes[0].set_title('Average Log Returns by Hour of Day')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Mean Log Return')
axes[0].grid(True, alpha=0.3)

# Return volatility by hour
axes[1].bar(hourly_returns['Hour'], hourly_returns['Return_Std'])
axes[1].set_title('Return Volatility by Hour of Day')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Return Standard Deviation')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print return statistics by hour
print("\nReturn Analysis by Hour:")
for _, row in hourly_returns.iterrows():
    print(f"Hour {row['Hour']:02.0f}:00 - Mean Return: {row['Mean_Return']:8.6f}, "
          f"Return Std: {row['Return_Std']:8.6f}, "
          f"Observations: {row['Return_Count']:4.0f}")

# Analyze bid-ask spread if available
if 'Bid Volume' in vx_data.columns and 'Ask Volume' in vx_data.columns:
    print("\nBid-Ask Analysis:")
    vx_data['Bid_Ask_Ratio'] = vx_data['Bid Volume'] / (vx_data['Bid Volume'] + vx_data['Ask Volume'])
    hourly_bid_ask = vx_data.groupby('Hour')['Bid_Ask_Ratio'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(10, 5))
    plt.bar(hourly_bid_ask['Hour'], hourly_bid_ask['mean'])
    plt.title('Bid-Ask Volume Ratio by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Bid/(Bid+Ask) Ratio')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Bid-Ask Ratio by Hour:")
    for _, row in hourly_bid_ask.iterrows():
        print(f"Hour {row['Hour']:02.0f}:00 - Bid Ratio: {row['mean']:6.4f}, Std: {row['std']:6.4f}")

# ## Variance Gamma Model in NumPyro
# We model the log returns as increments of a Variance Gamma process.

# In[1]:

vix_data = pd.read_csv(
    'data/VXMQ25_FUT_CFE [C][M]  Daily #6_GraphData.txt',
    skipinitialspace=True
)
vix_data['DateTime'] = vix_data.apply(lambda row: parse_datetime(row['Date'], row['Time']), axis=1)
vix_data = vix_data.sort_values('DateTime')

vix_data["basis"] = vix_data["VIX Last"] - vix_data["Last"]
plt.figure(figsize=(10, 5))
plt.plot(vix_data["basis"])
plt.axhline(y=float(np.mean(vix_data["basis"])), color='r', linestyle='--')
plt.title('Basis')
plt.xlabel('Time')
plt.ylabel('Basis')
plt.show()

log_basis = vix_data["basis"].apply(np.log)
log_basis_returns = log_basis.diff().dropna().values
plt.figure(figsize=(10, 5))
sns.histplot(np.asarray(log_basis_returns), bins=50, kde=True)
plt.title('Log Basis Distribution')
plt.xlabel('Log Basis')
plt.show()

# In[4]:
### Heston-Type Stochastic Volatility Model with Jumps (Bates Model Approximation)

# The code implements a discrete-time Euler-Maruyama approximation to the Bates model, which extends the Heston stochastic volatility model by adding Merton-style compound Poisson jumps to the returns process. This is suitable for modeling log-returns \( y_t = \log(S_t / S_{t-1}) \), assuming a time step \(\Delta t = 1\) (e.g., daily data). The model captures volatility clustering via persistence in the latent variance process \( v_t \), with correlation between return and volatility innovations.
# 
# #### Continuous-Time Reference Model (for Context)
# The underlying continuous-time dynamics (under the physical measure) are:
# \[
# dS_t / S_t = \mu \, dt + \sqrt{v_t} \, dW_t^{(1)} + dJ_t,
# \]
# \[
# dv_t = \kappa (\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW_t^{(2)},
# \]
# where:
# - \( \mu \): drift rate,
# - \( \kappa > 0 \): mean reversion speed,
# - \( \theta > 0 \): long-run variance,
# - \( \xi > 0 \): volatility of volatility,
# - \( \rho \): correlation, i.e., \( \mathbb{E}[dW_t^{(1)} dW_t^{(2)}] = \rho \, dt \),
# - \( J_t \): compound Poisson jump process with intensity \( \lambda \), each jump size \( Z \sim \mathcal{N}(\mu_j, \sigma_j^2) \).
# 
# This preserves the martingale property after drift adjustment (under risk-neutral measure, \(\mu = r - q - \lambda (\exp(\mu_j + \sigma_j^2/2) - 1)\), but here we estimate under physical measure).
# 
# #### Discrete-Time Model (as Implemented)
# For \( t = 1 \) to \( T \):
# - Start with latent variance \( v_1 = v_0 > 0 \) (sampled from Gamma prior).
# - Log-return observation:
#   \[
#   y_t = \mu - \frac{v_t}{2} + \sqrt{v_t} \, \epsilon_t^x + J_t,
#   \]
#   where \( \epsilon_t^x \sim \mathcal{N}(0, 1) \) (diffusion innovation), and \( J_t = \sum_{k=1}^{N_t} Z_k \) with \( N_t \sim \text{Poisson}(\lambda) \), \( Z_k \iid \mathcal{N}(\mu_j, \sigma_j^2) \).
# - Variance update:
#   \[
#   v_{t+1} = v_t + \kappa (\theta - v_t) + \xi \sqrt{v_t} \, \epsilon_t^v, \quad v_{t+1} = \max(v_{t+1}, 10^{-6}),
#   \]
#   where \( \epsilon_t^v = \rho \, \epsilon_t^x + \sqrt{1 - \rho^2} \, \epsilon_t^{\text{ind}} \), and \( \epsilon_t^{\text{ind}} \sim \mathcal{N}(0, 1) \) independent.
# 
# The jumps \( J_t \) are marginalized out for tractability (summing over \( n = 0 \) to \( \text{max_jumps} = 5 \)), avoiding sampling discrete latents.
# 
# #### Likelihood and Marginalization
# Given \( v_t \), the conditional distribution of \( y_t \) (marginalizing jumps) is computed as:
# - For each possible jump count \( n = 0, 1, \dots, 5 \):
#   \[
#   \tilde{\mu}_{t,n} = \mu - \frac{v_t}{2} + n \mu_j, \quad \tilde{\sigma}_{t,n}^2 = v_t + n \sigma_j^2.
#   \]
# - Log-probability contribution:
#   \[
#   \log p(y_t \mid v_t) = \log \sum_{n=0}^{5} \mathcal{N}(y_t \mid \tilde{\mu}_{t,n}, \tilde{\sigma}_{t,n}) \cdot \text{Poisson}(n \mid \lambda),
#   \]
#   implemented via `logsumexp` for numerical stability. This approximates the infinite sum (valid if \( \lambda \) is small, as \( P(n > 5) \) is negligible).
# 
# The factor `numpyro.factor("obs_y", log_prob_y)` adds this to the joint log-density for MCMC.
# 
# #### Innovation Approximation and Sequential Update
# To incorporate correlation \( \rho \), the return innovation is approximated using the observed \( y_t \) (which includes jumps):
# \[
# \hat{\epsilon}_t^x = \frac{y_t - (\mu - v_t / 2)}{\sqrt{v_t}}.
# \]
# This ignores jumps in backing out \( \epsilon_t^x \), a common filtering approximation (exact inference would require particle filtering or joint sampling of jumps). Then:
# - Sample \( \epsilon_t^{\text{ind}} \sim \mathcal{N}(0, 1) \),
# - Compute \( \epsilon_t^v = \rho \, \hat{\epsilon}_t^x + \sqrt{1 - \rho^2} \, \epsilon_t^{\text{ind}} \),
# - Update \( v_{t+1} \) as above.
# 
# This sequential structure (variance path dependence) necessitates `scan` for efficient compilation in JAX/NumPyro.
# 
# #### Priors (as in Code)
# - \( \mu \sim \mathcal{N}(0, 0.1) \),
# - \( \kappa \sim \text{Gamma}(2, 0.5) \) (encourages positivity),
# - \( \theta \sim \text{Gamma}(2, 10) \),
# - \( \xi \sim \text{Gamma}(2, 5) \),
# - \( \rho \sim \text{Uniform}(-1, 1) \),
# - \( v_0 \sim \text{Gamma}(2, 10) \),
# - \( \lambda = \exp(\log \lambda \sim \mathcal{N}(0, 0.1)) \),
# - \( \mu_j \sim \mathcal{N}(0, 0.1) \),
# - \( \sigma_j = \exp(\log \sigma_j \sim \mathcal{N}(0, 0.1)) \).
# 
# These are weakly informative, ensuring positivity where required.
# 
# #### Inference
# NUTS MCMC samples the parameters and latent \( \{\epsilon_t^{\text{ind}}\}_{t=1}^T \) (which imply the \( v_t \) path via deterministic updates). Jumps are marginalized, not sampled.
# 
# This setup balances tractability and realism for volatility clustering and jumps in returns data. For evidence on the approximation's validity, it aligns with methods in papers like Eraker (2004) on MCMC for stochastic volatility with jumps, where similar discretizations and innovation approximations are used for daily data.

# In[2]: Model
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
from jax.scipy.special import logsumexp

def heston_jump_model(data, max_jumps=1):
    # Data is already chunked: shape (num_chunks, chunk_size)
    data = jnp.asarray(data)
    num_chunks, chunk_size = data.shape

    # Heston parameters (assuming dt=1 for simplicity)
    mu = numpyro.sample("mu", dist.Normal(-0.01, 0.001))  # Drift
    kappa = numpyro.sample("kappa", dist.Gamma(2, 125))  # Mean reversion speed, prior for positive
    xi = numpyro.sample("xi", dist.Gamma(3, 1500))  # Vol of vol
    rho_raw = numpyro.sample("rho_raw", dist.Beta(1.5, 1.5))  # Correlation
    rho = numpyro.deterministic("rho", 2 * rho_raw - 1)  # Convert to [-1, 1]
    theta = numpyro.sample("theta", dist.Gamma(2, 200))  # Long-run variance

    # Jump parameters (kept similar)
    log_lambda_jump = numpyro.sample("log_lambda_jump", dist.Normal(-0.5, 0.1))
    lambda_jump = numpyro.deterministic("lambda_jump", jnp.exp(log_lambda_jump))
    mu_jump = numpyro.sample("mu_jump", dist.Normal(0.2, 0.05))
    log_sigma_jump = numpyro.sample("log_sigma_jump", dist.Normal(-1.2, 0.1))
    sigma_jump = numpyro.deterministic("sigma_jump", jnp.exp(log_sigma_jump))

    jump_compensator = jnp.clip(lambda_jump * (jnp.exp(mu_jump + sigma_jump**2 / 2) - 1), -0.1, 0.1)
    mu_Q = numpyro.deterministic("mu_Q", mu)  # - jump_compensator)

    # Precompute jump-related terms (scalar across chunks)
    jump_counts = jnp.arange(max_jumps + 1)
    poisson_log_probs = dist.Poisson(lambda_jump).log_prob(jump_counts)
    jump_means = jump_counts * mu_jump
    jump_vars = jump_counts * sigma_jump**2

    # Initial variance per chunk (independent across chunks)
    v = numpyro.sample("v0", dist.Gamma(2, 5).expand([num_chunks]).to_event(1))

    # Loop over time steps (vectorized over chunks)
    for t in range(chunk_size):
        y_obs = data[:, t]  # (num_chunks,)
        is_observed = ~jnp.isnan(y_obs)
        y_masked = jnp.where(is_observed, y_obs, 0.)  # Replace NaN with 0 to avoid NaN in log_prob

        mean_y = mu_Q - v / 2  # (num_chunks,)
        std_y = jnp.sqrt(v + 1e-8)  # (num_chunks,)

        # Marginalize over jumps (vectorized over chunks and jumps)
        total_means = mean_y[:, None] + jump_means[None, :]  # (num_chunks, max_jumps+1)
        total_vars = v[:, None] + jump_vars[None, :]  # (num_chunks, max_jumps+1)
        total_stds = jnp.sqrt(total_vars + 1e-8)
        log_probs = dist.Normal(total_means, total_stds).log_prob(y_masked[:, None]) + poisson_log_probs[None, :]
        log_prob_y = logsumexp(log_probs, axis=1)  # (num_chunks,)

        # Masked likelihood (skip for NaN/padding); sum over chunks
        numpyro.factor(f"obs_y_{t}", jnp.where(is_observed, log_prob_y, 0.).sum())

        # Sample independent part always (batched over chunks)
        eps_ind = numpyro.sample(f"eps_ind_{t}", dist.Normal(0, 1).expand([num_chunks]).to_event(1))

        # For eps_x: use observed (approx) if available, else sample from prior
        eps_x_missing = numpyro.sample(f"eps_x_missing_{t}", dist.Normal(0, 1).expand([num_chunks]).to_event(1))
        eps_x_obs = (y_masked - mean_y) / std_y
        eps_x = jnp.where(is_observed, eps_x_obs, eps_x_missing)

        # Volatility innovation (full var=1 preserved in both cases)
        eps_v = rho * eps_x + jnp.sqrt(1 - rho**2) * eps_ind

        mean_v = v + kappa * (theta - v)
        std_v = xi * jnp.sqrt(v + 1e-8)
        v_next = mean_v + std_v * eps_v
        v = jnp.maximum(v_next, 1e-6)


# In[13] Model with y
def heston_jump_model_with_y(time_steps, max_jumps=1, num_chunks=1):
    # No data needed; time_steps is the number of prediction steps

    # Heston parameters (assuming dt=1 for simplicity)
    mu = numpyro.sample("mu", dist.Normal(-0.01, 0.001))  # Drift
    kappa = numpyro.sample("kappa", dist.Gamma(2, 125))  # Mean reversion speed, prior for positive
    xi = numpyro.sample("xi", dist.Gamma(3, 1500))  # Vol of vol
    rho_raw = numpyro.sample("rho_raw", dist.Beta(1.5, 1.5))  # Correlation
    rho = numpyro.deterministic("rho", 2 * rho_raw - 1)  # Convert to [-1, 1]
    theta = numpyro.sample("theta", dist.Gamma(2, 200))  # Long-run variance

    # Jump parameters (kept similar)
    log_lambda_jump = numpyro.sample("log_lambda_jump", dist.Normal(-0.5, 0.1))
    lambda_jump = numpyro.deterministic("lambda_jump", jnp.exp(log_lambda_jump))
    mu_jump = numpyro.sample("mu_jump", dist.Normal(0.2, 0.05))
    log_sigma_jump = numpyro.sample("log_sigma_jump", dist.Normal(-1.2, 0.1))
    sigma_jump = numpyro.deterministic("sigma_jump", jnp.exp(log_sigma_jump))

    jump_compensator = jnp.clip(lambda_jump * (jnp.exp(mu_jump + sigma_jump**2 / 2) - 1), -0.1, 0.1)
    mu_Q = numpyro.deterministic("mu_Q", mu)  # - jump_compensator)

    # Initial variance per chunk
    v = numpyro.sample("v0", dist.Gamma(2, 5).expand([num_chunks]).to_event(1))

    y_pred_list = []

    # Loop for predictions (vectorized over chunks)
    for t in range(time_steps):
        mean_y = mu_Q - v / 2
        std_y = jnp.sqrt(v + 1e-8)

        jump_count = numpyro.sample(f"jump_count_{t}", dist.Poisson(lambda_jump).expand([num_chunks]).to_event(1))
        jump_mean = jump_count * mu_jump
        jump_var = jump_count * sigma_jump**2
        std_full = jnp.sqrt(v + jump_var + 1e-8)

        y_t = numpyro.sample(f"y_pred_{t}", dist.Normal(mean_y + jump_mean, std_full).to_event(1))
        y_pred_list.append(y_t)

        # Approximation for innovation (consistent with model)
        eps_x = (y_t - mean_y) / std_y
        eps_ind = numpyro.sample(f"eps_ind_{t}", dist.Normal(0, 1).expand([num_chunks]).to_event(1))
        eps_v = rho * eps_x + jnp.sqrt(1 - rho**2) * eps_ind

        mean_v = v + kappa * (theta - v)
        std_v = xi * jnp.sqrt(v + 1e-8)
        v_next = mean_v + std_v * eps_v
        v = jnp.maximum(v_next, 1e-6)

    # Stack predictions into single site (shape: num_chunks x time_steps)
    y_pred = jnp.stack(y_pred_list, axis=1)
    numpyro.deterministic("y_pred", y_pred)

# In[10] SVI
import multiprocessing

print(multiprocessing.cpu_count())
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())

from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal  # Or AutoDelta for mean-field
from jax.random import PRNGKey
import numpyro.optim as optim

numpyro.set_platform('cpu')
device_num = 4
numpyro.set_host_device_count(device_num)
print(jax.local_device_count())

guide = AutoDiagonalNormal(heston_jump_model)  # Flexible; captures correlations diagonally

# SVI setup
svi = SVI(
    model=heston_jump_model,  # Or reparam_model if using
    guide=guide,
    optim=optim.ClippedAdam(step_size=0.001, clip_norm=1.0),  # Learning rate; tune if needed (0.001-0.01)
    loss=Trace_ELBO(num_particles=10)  # Higher particles for stability
)

log_returns_jax = jnp.array(log_returns_chunked)
# Run SVI
svi_result = svi.run(
    PRNGKey(0),
    num_steps=10**4,  # Increase to 10k if needed; converges fast
    data=log_returns_jax,
)

# Extract params (approx posterior means)
params = svi_result.params


# In[74]: SVI Predictive
import jax.random as random
from numpyro.infer import Predictive
import matplotlib.pyplot as plt
import jax.numpy as jnp
from numpyro.diagnostics import hpdi

num_days = 15
num_samples = 1000  # Increase for better distribution estimate
current_price = 18.0  # Example starting price; replace with actual current price

# Generate predictive samples
predictive = Predictive(heston_jump_model_with_y, guide=guide, params=params, num_samples=num_samples, return_sites=["y_pred"])
predictive_samples = predictive(random.PRNGKey(np.random.randint(1000000)), time_steps=num_days)
y_pred_samples = predictive_samples["y_pred"]  # Shape: (num_samples, num_chunks, num_days); assuming num_chunks=1 for plotting

# Compute cumulative log-returns
cum_log_returns = jnp.cumsum(y_pred_samples, axis=-1)  # Shape: (num_samples, num_chunks, num_days)

# Compute price paths (real returns via exponentiation, starting from current_price)
price_paths = current_price * jnp.exp(cum_log_returns)

# Plot price paths (squeeze if num_chunks=1)
num_paths_to_plot = min(10, num_samples)  # Plot up to 10 paths for clarity
plt.figure(figsize=(12, 6))
time = jnp.arange(num_days)
for i in range(num_paths_to_plot):
    path = price_paths[i, 0] if price_paths.shape[1] == 1 else price_paths[i].mean(axis=0)  # Average over chunks if multiple
    plt.plot(time, path, label=f'Path {i+1}', alpha=0.7)

plt.title('Simulated Price Paths from Bates Model')
plt.xlabel('Time Step (Days)')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Extract final prices (at end of time steps)
final_prices = price_paths[:, 0, -1] if price_paths.shape[1] == 1 else price_paths[:, :, -1].mean(axis=1)  # Average over chunks if multiple

# Compute HPDI (89% by default)
hpdi_interval = hpdi(final_prices, prob=0.89)

# Visualize distribution of final prices with HPDI
plt.figure(figsize=(10, 5))
sns.histplot(final_prices, bins=30, kde=True, stat='density')
plt.axvline(hpdi_interval[0], color='r', linestyle='--', label=f'HPDI Lower: {hpdi_interval[0]:.2f}')
plt.axvline(hpdi_interval[1], color='r', linestyle='--', label=f'HPDI Upper: {hpdi_interval[1]:.2f}')
plt.title('Distribution of Final Prices with 89% HPDI')
plt.xlabel('Final Price')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# In[11] Loss Curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assume svi_result is from svi.run(...)
losses = svi_result.losses  # Array of -ELBO values

# Plot
plt.figure(figsize=(10, 6))
plt.plot(losses, label='SVI Loss (-ELBO)')
plt.xlabel('Iteration')
plt.ylabel('Loss (-ELBO)')
plt.title('SVI Loss Curve for Bates Model')
plt.axhline(0, color='red', linestyle='--', label='Zero Line')  # Highlight negative loss
plt.legend()
plt.grid(True)
plt.show()

# In[10] Parameter means
import jax.numpy as jnp
import jax.random as random
from numpyro.infer import Predictive

# Assume SVI has been run: svi_result = svi.run(...)
# and guide is defined (e.g., AutoDiagonalNormal(heston_jump_model))
params = svi_result.params

# Sample from posterior
rng_key = random.PRNGKey(1)  # Reproducible seed
num_samples = 2000
posterior_samples = guide.sample_posterior(rng_key, params, sample_shape=(num_samples,))

# List of stochastic parameters
stochastic_params = ['mu', 'kappa', 'xi', 'rho_raw', 'theta', 'v0', 'log_lambda_jump', 'mu_jump', 'log_sigma_jump']

# Compute and print means for stochastic parameters
print("Mean of each stochastic parameter:")
for param in stochastic_params:
    mean = jnp.mean(posterior_samples[param])
    print(f"{param}: {mean:.4f}")

# Compute deterministic sites from posterior samples
rho = 2 * posterior_samples['rho_raw'] - 1
lambda_jump = jnp.exp(posterior_samples['log_lambda_jump'])
sigma_jump = jnp.exp(posterior_samples['log_sigma_jump'])
jump_compensator = jnp.clip(lambda_jump * (jnp.exp(posterior_samples['mu_jump'] + sigma_jump**2 / 2) - 1), -0.1, 0.1)
mu_Q = posterior_samples['mu']  # As per model; adjust if using compensator

# List of deterministic parameters
deterministic_params = {
    'rho': rho,
    'lambda_jump': lambda_jump,
    'sigma_jump': sigma_jump,
    'jump_compensator': jump_compensator,
    'mu_Q': mu_Q
}

# Compute and print means for deterministic parameters
print("\nMean of each deterministic parameter:")
for param, values in deterministic_params.items():
    mean = jnp.mean(values)
    print(f"{param}: {mean:.4f}")

# Use predictive model (from earlier) for y_pred
predictive = Predictive(heston_jump_model_with_y, guide=guide, params=params, num_samples=100, return_sites=["y_pred"])
predictive_samples = predictive(random.PRNGKey(1), time_steps=23)
y_pred_std = jnp.std(predictive_samples["y_pred"], axis=0).mean()
print(f"Predicted log-returns std: {y_pred_std:.4f} (Data std: 0.037)")

# Check data skew
import jax.numpy as jnp
data_skew = jnp.mean((log_returns_jax - jnp.mean(log_returns_jax))**3) / (jnp.std(log_returns_jax)**3)
print(f"Data skew: {data_skew:.4f}")


# In[11] Posterior Plots
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp

rng_key = PRNGKey(1)
num_samples = 2000
posterior_samples = guide.sample_posterior(rng_key, params, sample_shape=(num_samples,))

# List of stochastic parameters (adjust if needed based on your model)
stochastic_params = ['mu', 'kappa', 'xi', 'rho_raw', 'theta', 'log_lambda_jump', 'mu_jump', 'log_sigma_jump']

# Compute deterministic parameters from stochastic samples
lambda_jump_samples = jnp.exp(posterior_samples['log_lambda_jump'])
rho_samples = 2 * posterior_samples['rho_raw'] - 1
sigma_jump_samples = jnp.exp(posterior_samples['log_sigma_jump'])

# Combine all parameters for plotting (stochastic + deterministic)
all_params = {
    'mu': posterior_samples['mu'],
    'kappa': posterior_samples['kappa'],
    'xi': posterior_samples['xi'],
    'rho_raw': posterior_samples['rho_raw'],
    'theta': posterior_samples['theta'],
    'log_lambda_jump': posterior_samples['log_lambda_jump'],
    'mu_jump': posterior_samples['mu_jump'],
    'log_sigma_jump': posterior_samples['log_sigma_jump'],
    'lambda_jump': lambda_jump_samples,
    'rho': rho_samples,
    'sigma_jump': sigma_jump_samples
}

# Create subplots: 4x3 grid for 12 params
fig, axes = plt.subplots(4, 3, figsize=(18, 24))
axes = axes.flatten()  # Flatten for easy indexing

for i, (param, samples) in enumerate(all_params.items()):
    sns.histplot(samples, kde=True, ax=axes[i])
    axes[i].set_title(f'Posterior for {param}')
    axes[i].set_xlabel(param)
    axes[i].set_ylabel('Density')

plt.tight_layout()
plt.show()

# In[12]
import arviz as az
inf_data = az.from_dict(posterior_samples)
az.plot_posterior(inf_data)
az.summary(inf_data)


# In[14]
predictive = Predictive(heston_jump_model_with_y, guide=guide, params=params, num_samples=2000)
predictive_samples = predictive(PRNGKey(1), data=log_returns_jax, time_steps=len(log_returns_jax))
print(np.exp(predictive_samples["y_pred"]))

# In[15]
# Prepare data for JAX
log_returns_jax = jnp.array(log_returns)
#config = {
#    "v0": TransformReparam(),
#    "theta": TransformReparam()
#}
#reparam_model = numpyro.handlers.reparam(heston_jump_model, config=config)

nuts_kernel = NUTS(heston_jump_model, adapt_step_size=True, target_accept_prob=0.9, dense_mass=True, max_tree_depth=8)
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=device_num)
rng_key = jax.random.PRNGKey(0)
rng_key, rng_key_ = jax.random.split(rng_key)
mcmc.run(rng_key_, data=log_returns_jax, time_steps=len(log_returns_jax))
mcmc.print_summary()
posterior_samples = mcmc.get_samples()


# In[70]:
import os
import multiprocessing

print(multiprocessing.cpu_count())
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.control_flow import scan
import jax
from numpyro.infer.reparam import LocScaleReparam

#numpyro.set_platform('gpu')
numpyro.set_platform('cpu')
device_num = 1
numpyro.set_host_device_count(device_num)
print(jax.local_device_count())

def vg_jump_model(data, time_steps, max_jumps=5):
    # Ensure data shape
    data = jnp.asarray(data)
    assert data.shape == (time_steps,), f"Expected data shape ({time_steps},), got {data.shape}"

    # Reparameterized Variance Gamma parameters
    nu = numpyro.sample("nu", dist.Gamma(4, 15))
    #log_nu = numpyro.sample("log_nu", dist.Normal(0, 0.5))  # nu ~ log-normal, mean ~1
    #nu = numpyro.deterministic("nu", jnp.exp(log_nu))
    theta = numpyro.sample("theta", dist.Normal(-0.2, 0.02))  # Tighter prior
    log_sigma = numpyro.sample("log_sigma", dist.Normal(0, 0.1))  # sigma ~ log-normal, mean ~0.2
    sigma = numpyro.deterministic("sigma", jnp.exp(log_sigma))

    # Reparameterized Jump parameters
    log_lambda_jump = numpyro.sample("log_lambda_jump", dist.Normal(0, 0.1))  # lambda_jump ~ log-normal, mean ~1
    lambda_jump = numpyro.deterministic("lambda_jump", jnp.exp(log_lambda_jump))
    mu_jump = numpyro.sample("mu_jump", dist.Normal(0, 0.1))  # Scalar
    log_sigma_jump = numpyro.sample("log_sigma_jump", dist.Normal(0, 0.1))  # sigma_jump ~ log-normal, mean ~0.2
    sigma_jump = numpyro.deterministic("sigma_jump", jnp.exp(log_sigma_jump))

    # Learnable observation variance
    #log_obs_std = numpyro.sample("log_obs_std", dist.Normal(0, 0.1))  # obs_std ~ log-normal, mean ~1
    #obs_std = numpyro.deterministic("obs_std", jnp.exp(log_obs_std))

    # Non-centered Variance Gamma: Gamma subordinator
    G_raw = numpyro.sample("G_raw", dist.Gamma(1, 1).expand([time_steps]))  # Standard Gamma
    G = numpyro.deterministic("G", G_raw * (1 / nu))  # Scale by 1/nu
    mean = numpyro.deterministic("mean", theta * G)  # Shape: (time_steps,)
    std = numpyro.deterministic("std", sigma * jnp.sqrt(G))  # Shape: (time_steps,)

    # VG increments
    vg_increment = numpyro.sample(
        "vg_increment",
        dist.Normal(mean, std).expand([time_steps])  # Shape: (time_steps,)
    )

    # Marginalize Poisson jumps
    jump_counts = jnp.arange(max_jumps + 1)  # Shape: (max_jumps+1,)
    poisson_log_probs = dist.Poisson(lambda_jump).log_prob(jump_counts)  # Shape: (max_jumps+1,)

    # Compute log-likelihood per time step
    def log_likelihood_per_step(t):
        jump_sizes = jump_counts * mu_jump  # Shape: (max_jumps+1,)
        total_mean = vg_increment[t] + jump_sizes  # Shape: (max_jumps+1,)
        total_std = jnp.sqrt(sigma_jump**2)  # Scalar
        log_probs = dist.Normal(total_mean, total_std).log_prob(data[t])  # Shape: (max_jumps+1,)
        return logsumexp(log_probs + poisson_log_probs)  # Scalar

    # Vectorize over time steps
    log_likelihoods = jnp.vectorize(log_likelihood_per_step, signature='()->()')(jnp.arange(time_steps))  # Shape: (time_steps,)
    for t in range(time_steps):
        numpyro.factor(f"log_prob_{t}", log_likelihoods[t])


# Prepare data for JAX
log_returns_jax = jnp.array(log_returns)

#config = {"G": LocScaleReparam(centered=0)}
#better_model = numpyro.handlers.reparam(vg_model, config=config)
nuts_kernel = NUTS(vg_jump_model, adapt_step_size=True, target_accept_prob=0.85)
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=device_num)
rng_key = jax.random.PRNGKey(0)
rng_key, rng_key_ = jax.random.split(rng_key)
mcmc.run(rng_key_, data=log_returns_jax, time_steps=len(log_returns_jax))
mcmc.print_summary()
posterior_samples = mcmc.get_samples()


# ## Posterior Plots
# Let's visualize the posterior distributions of the VG parameters.

# In[71]:
import arviz as az
# Plot only the main VG parameters, excluding the G array
main_params = {k: v for k, v in posterior_samples.items() if k in ["nu", "sigma", "lambda_jump", "mu_jump", "sigma_jump", "theta"]}
# Convert to InferenceData format for arviz
idata = az.from_dict(posterior=main_params)
az.plot_trace(data)
plt.tight_layout()
plt.show()

# In[72]:



# In[75]:
def vg_jump(nu, theta, sigma, lambda_jump, mu_jump, sigma_jump, timesteps):
    # Simulate Gamma subordinator
    G = np.random.gamma(1 / nu, 1 / nu, timesteps)
    # VG increments
    mean = theta * G
    std = sigma * np.sqrt(G)
    vg_increments = np.random.normal(mean, std)
    # Poisson jumps per time step
    num_jumps = np.random.poisson(lambda_jump, timesteps)
    # Jump sizes, zero when num_jumps = 0
    jump_sizes = np.random.normal(mu_jump, sigma_jump, timesteps) * (num_jumps > 0) * num_jumps
    # Total log returns
    total_log_returns = vg_increments + jump_sizes
    # Convert to price or basis path
    cumulative = np.cumsum(total_log_returns)
    return cumulative 

nu, theta, sigma, lambda_jump, mu_jump, sigma_jump = posterior_samples['nu'].mean(), posterior_samples['theta'].mean(), posterior_samples['sigma'].mean(), posterior_samples['lambda_jump'].mean(), posterior_samples['mu_jump'].mean(), posterior_samples['sigma_jump'].mean()
# nu, sigma, theta = 0.8, 0.2, -0.01
timesteps = 10
start_price = 18.76
plt.figure(figsize=(12, 6))
for i in range(10):
    y_new = vg_jump(nu, theta, sigma, lambda_jump, mu_jump, sigma_jump, timesteps)
    plt.plot(y_new, alpha=0.7, label=f'Path {i+1}')
plt.title('Simulated Variance Gamma Paths')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.grid(True, alpha=0.3)
plt.show()


# In[76]:

# Plot cumulative distribution at the last time step
last_step_values = y_new[:, -1]  # Get values at the last time step

plt.figure(figsize=(10, 6))
plt.hist(last_step_values, bins=30, density=True, alpha=0.7, label='Simulated')
plt.xlabel('Cumulative Log Return')
plt.ylabel('Density')
plt.title('Cumulative Distribution at Final Time Step')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print some statistics
print(f"Mean: {np.mean(last_step_values):.4f}")
print(f"Std: {np.std(last_step_values):.4f}")
print(f"Min: {np.min(last_step_values):.4f}")
print(f"Max: {np.max(last_step_values):.4f}")


# In[75]:
# Plot empirical CDF of the ending paths distribution
from scipy import stats

plt.figure(figsize=(10, 6))

# Sort the values for CDF plotting
sorted_values = np.sort(last_step_values)
n = len(sorted_values)
cdf = np.arange(1, n + 1) / n

plt.plot(sorted_values, cdf, 'b-', linewidth=2, label='Empirical CDF')
plt.xlabel('Cumulative Log Return at Final Time Step')
plt.ylabel('Cumulative Probability')
plt.title('Empirical CDF of Simulated Path Endpoints')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Also plot against theoretical normal CDF for comparison
mean_val = np.mean(last_step_values)
std_val = np.std(last_step_values)
x_normal = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 1000)
normal_cdf = stats.norm.cdf(x_normal, mean_val, std_val)

plt.figure(figsize=(10, 6))
plt.plot(sorted_values, cdf, 'b-', linewidth=2, label='Empirical CDF (VG)')
plt.plot(x_normal, normal_cdf, 'r--', linewidth=2, label='Normal CDF')
plt.xlabel('Cumulative Log Return at Final Time Step')
plt.ylabel('Cumulative Probability')
plt.title('Empirical CDF vs Normal CDF')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# %%

# ## Analysis of Long-Term VG Path Simulation
# Ideas at this point:
# - Sampling from the posterior of each parameter allows for an understanding of full distribution of paths and should show us realistic paths
# - Otherwise we are just looking at the mean path within sample space.
# - What is our measure of good? When do we know we have done better?
# - Conditioning the theta, sigma, and nu on the current environment will allow us to be more accurate and "better" when estimating the posterior cdf at the end of paths.
# - Consider the risk neutral vs risk estimates of the paths and their difference with the spot to estimate basis in the case of no predictive power.
# - What is the edge we can uncover above and beyond the risk-neutral basis estimate?
# - Determine the optimal hedge ratio and factor that into the path estimates to uncover the true risk.


# %%