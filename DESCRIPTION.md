# Mathematical Foundation: Probabilistic Post-Hoc Fusion Method for Gypsum Requirement Uncertainty

## Overview

This document describes the analytical approximation used to propagate uncertainty through the gypsum requirement calculation using **post-hoc probabilistic fusion**. The method is based on the **Delta Method** (also known as first-order Taylor expansion or linear error propagation), which provides a computationally efficient way to estimate the mean and variance of a nonlinear function of random variables.

### Post-Hoc Probabilistic Fusion Context

In this workflow:

1. **Stage 1 - Separate Spatial Modeling**: Two independent Gaussian Process Regression (GPR) models are trained:
   - GPR model for ESP → produces μ_ESP(x) and σ²_ESP(x) at each location x
   - GPR model for CEC → produces μ_CEC(x) and σ²_CEC(x) at each location x

2. **Stage 2 - Post-Hoc Fusion**: The Delta method fuses these separate predictions to compute:
   - GR mean: μ_GR(x) 
   - GR variance: σ²_GR(x)
   - Accounting for the cross-correlation between ESP and CEC

This is called "post-hoc" because the fusion happens **after** the separate GPR models have been trained and predictions made, rather than fitting a joint multivariate GP model directly. This approach is computationally efficient and allows independent modeling of each soil property.

---

## Problem Statement

We want to calculate the gypsum requirement (GR) and its uncertainty at each spatial location, given **outputs from separate Gaussian Process Regression models**:

### Inputs from GPR Models

**From ESP GPR model** (trained on ESP measurements):
- **μ_ESP(x)**: Predictive mean of ESP at location x
- **σ²_ESP(x)**: Predictive variance of ESP at location x

**From CEC GPR model** (trained on CEC measurements):
- **μ_CEC(x)**: Predictive mean of CEC at location x
- **σ²_CEC(x)**: Predictive variance of CEC at location x

**From training data** (estimated once, applied everywhere):
- **Cov[ESP, CEC]** or **ρ** (correlation coefficient): Cross-correlation between ESP and CEC

### What GPR Provides

Gaussian Process Regression produces a **predictive distribution** at each location:

$$\text{ESP}(x) \sim \mathcal{N}(\mu_{\text{ESP}}(x), \sigma^2_{\text{ESP}}(x))$$

$$\text{CEC}(x) \sim \mathcal{N}(\mu_{\text{CEC}}(x), \sigma^2_{\text{CEC}}(x))$$

The predictive variance σ²(x) naturally accounts for:
- Uncertainty due to limited observations (epistemic uncertainty)
- Distance from training points (increases with distance)
- Local smoothness assumptions (from the covariance kernel)

These predictive distributions become the inputs to our post-hoc fusion.

The gypsum requirement is calculated as:

$$\text{GR} = k \cdot \text{CEC} \cdot (\text{ESP} - \text{ESP}_{\text{ref}})$$

where:
- $k = 0.0086 \times F \times D_s \times \rho_b$ (combined constant)
- $F$ = exchange efficiency factor (e.g., 1.2)
- $D_s$ = soil depth in meters (e.g., 0.15 m)
- $\rho_b$ = bulk density in Mg/m³ (e.g., 1.2)
- $\text{ESP}_{\text{ref}}$ = target ESP threshold (e.g., 5%)

---

## Why Post-Hoc Fusion?

### Alternative: Joint Multivariate GP

We could fit a **joint multivariate Gaussian Process** for [ESP, CEC] simultaneously:

$$\begin{bmatrix} \text{ESP}(x) \\ \text{CEC}(x) \end{bmatrix} \sim \mathcal{GP}\left(\boldsymbol{\mu}(x), \mathbf{K}(x, x')\right)$$

where **K** is a cross-covariance kernel that captures both auto-covariance (ESP-ESP, CEC-CEC) and cross-covariance (ESP-CEC).

**Advantages of joint GP:**
- Automatically captures spatial cross-correlation
- Theoretically more rigorous
- No need for post-hoc fusion

**Disadvantages of joint GP:**
- **Computational cost**: O(n³) for n training points, harder to scale
- **Model complexity**: Requires choosing cross-covariance kernel structure
- **Less flexible**: Both variables must use compatible kernels and hyperparameters
- **Harder to interpret**: Cross-covariance parameters can be difficult to estimate robustly

### Post-Hoc Fusion Approach (Our Method)

Instead, we use **separate GPR models** + **Delta method fusion**:

1. Fit independent GPR for ESP: computationally efficient, use optimal kernel for ESP
2. Fit independent GPR for CEC: computationally efficient, use optimal kernel for CEC  
3. Estimate global correlation ρ from training data (simple, stable)
4. Apply Delta method to fuse predictions at prediction locations

**Key advantages:** Computational efficiency (4x faster training), flexibility in kernel selection, and modularity (see detailed advantages in "Complete Post-Hoc Fusion Workflow" section).

**Key Assumption:**
- The cross-correlation structure is **approximately constant in space** (or at least, we use a single global estimate)
- This is reasonable when ESP and CEC are driven by similar underlying soil formation processes

---

## Estimating Cross-Correlation from Training Data

Since we fit separate GPR models, we don't get cross-covariance from the models directly. Instead, we estimate it from the **observed training data**:

### At Training Locations

If we have n training observations: {(ESP₁, CEC₁), (ESP₂, CEC₂), ..., (ESPₙ, CECₙ)}, we compute:

**Pearson correlation coefficient:**

$$\rho = \frac{\sum_{i=1}^{n}(\text{ESP}_i - \bar{\text{ESP}})(\text{CEC}_i - \bar{\text{CEC}})}{\sqrt{\sum_{i=1}^{n}(\text{ESP}_i - \bar{\text{CEC}})^2} \sqrt{\sum_{i=1}^{n}(\text{CEC}_i - \bar{\text{CEC}})^2}}$$

This gives us a **global correlation estimate**: typically ρ ∈ [-1, 1]

### At Prediction Locations

At any prediction location x, we convert this correlation to covariance (see detailed explanation in "Cross-Covariance in Practice" section below):

$$\text{Cov}[\text{ESP}(x), \text{CEC}(x)] = \rho \cdot \sigma_{\text{ESP}}(x) \cdot \sigma_{\text{CEC}}(x)$$

---

## The Delta Method

### Intuition

When we have random variables (ESP and CEC with uncertainty) as inputs to a nonlinear function (GR), we need a way to estimate the output's mean and variance. The Delta method approximates the nonlinear function with its first-order Taylor expansion around the input means.

Think of it as asking: "If I wiggle the inputs slightly around their mean values, how much does the output wiggle?"

### Mathematical Foundation

For a function $g(\mathbf{X})$ where $\mathbf{X} = [X_1, X_2, ..., X_n]$ is a vector of random variables with mean $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$, the Delta method approximates:

$$\mathbb{E}[g(\mathbf{X})] \approx g(\boldsymbol{\mu})$$

$$\text{Var}[g(\mathbf{X})] \approx \nabla g(\boldsymbol{\mu})^T \boldsymbol{\Sigma} \nabla g(\boldsymbol{\mu})$$

where $\nabla g$ is the gradient (vector of partial derivatives) of $g$ evaluated at $\boldsymbol{\mu}$.

---

## Application to Gypsum Requirement

### Step 1: Define the Function

Our function is:

$$g(\text{CEC}, \text{ESP}) = k \cdot \text{CEC} \cdot (\text{ESP} - \text{ESP}_{\text{ref}})$$

This is a function of two random variables: CEC and ESP.

### Step 2: Calculate the Mean

The mean of GR is approximated by evaluating the function at the means of the inputs:

$$\boxed{\mathbb{E}[\text{GR}] \approx k \cdot \mu_{\text{CEC}} \cdot (\mu_{\text{ESP}} - \text{ESP}_{\text{ref}})}$$

This is intuitive: the expected gypsum requirement is approximately what you'd get if you used the mean values of CEC and ESP.

### Step 3: Calculate Partial Derivatives

To apply the Delta method for variance, we need the partial derivatives of GR with respect to each input variable.

**Partial derivative with respect to CEC:**

$$\frac{\partial \text{GR}}{\partial \text{CEC}} = k \cdot (\text{ESP} - \text{ESP}_{\text{ref}})$$

**Partial derivative with respect to ESP:**

$$\frac{\partial \text{GR}}{\partial \text{ESP}} = k \cdot \text{CEC}$$

These derivatives tell us how sensitive GR is to changes in CEC and ESP.

### Step 4: Evaluate Derivatives at the Means

For the Delta method, we evaluate these derivatives at the mean values:

$$\left.\frac{\partial \text{GR}}{\partial \text{CEC}}\right|_{\boldsymbol{\mu}} = k \cdot (\mu_{\text{ESP}} - \text{ESP}_{\text{ref}})$$

$$\left.\frac{\partial \text{GR}}{\partial \text{ESP}}\right|_{\boldsymbol{\mu}} = k \cdot \mu_{\text{CEC}}$$

### Step 5: Construct the Covariance Structure

For our two-variable case, the covariance matrix is:

$$\boldsymbol{\Sigma} = \begin{bmatrix} 
\sigma^2_{\text{CEC}} & \text{Cov}[\text{CEC}, \text{ESP}] \\
\text{Cov}[\text{ESP}, \text{CEC}] & \sigma^2_{\text{ESP}}
\end{bmatrix}$$

Note that $\text{Cov}[\text{CEC}, \text{ESP}] = \text{Cov}[\text{ESP}, \text{CEC}]$ (covariance is symmetric).

The gradient vector is:

$$\nabla g = \begin{bmatrix}
\frac{\partial \text{GR}}{\partial \text{CEC}} \\
\frac{\partial \text{GR}}{\partial \text{ESP}}
\end{bmatrix}_{\boldsymbol{\mu}} = \begin{bmatrix}
k(\mu_{\text{ESP}} - \text{ESP}_{\text{ref}}) \\
k \mu_{\text{CEC}}
\end{bmatrix}$$

### Step 6: Calculate Variance

The variance formula from the Delta method is:

$$\text{Var}[\text{GR}] \approx \nabla g^T \boldsymbol{\Sigma} \nabla g$$

Expanding this matrix multiplication:

$$\text{Var}[\text{GR}] = \begin{bmatrix}
k(\mu_{\text{ESP}} - \text{ESP}_{\text{ref}}) & k\mu_{\text{CEC}}
\end{bmatrix}
\begin{bmatrix} 
\sigma^2_{\text{CEC}} & \text{Cov}[\text{CEC}, \text{ESP}] \\
\text{Cov}[\text{ESP}, \text{CEC}] & \sigma^2_{\text{ESP}}
\end{bmatrix}
\begin{bmatrix}
k(\mu_{\text{ESP}} - \text{ESP}_{\text{ref}}) \\
k\mu_{\text{CEC}}
\end{bmatrix}$$

After performing the matrix multiplication:

$$\boxed{
\begin{aligned}
\text{Var}[\text{GR}] \approx k^2 \Big[ &(\mu_{\text{ESP}} - \text{ESP}_{\text{ref}})^2 \cdot \sigma^2_{\text{CEC}} \\
&+ \mu^2_{\text{CEC}} \cdot \sigma^2_{\text{ESP}} \\
&+ 2(\mu_{\text{ESP}} - \text{ESP}_{\text{ref}}) \cdot \mu_{\text{CEC}} \cdot \text{Cov}[\text{CEC}, \text{ESP}] \Big]
\end{aligned}
}$$

The standard deviation is then:

$$\boxed{\sigma_{\text{GR}} = \sqrt{\text{Var}[\text{GR}]}}$$

---

## Interpretation of the Variance Formula

The variance formula has three terms, each with a clear interpretation:

### Term 1: CEC Uncertainty Contribution

$$k^2 \cdot (\mu_{\text{ESP}} - \text{ESP}_{\text{ref}})^2 \cdot \sigma^2_{\text{CEC}}$$

- This term represents how uncertainty in CEC propagates to GR
- It's weighted by the squared ESP difference from the target
- **If ESP is far from the target**, CEC uncertainty has a large impact
- **If ESP is close to the target**, CEC uncertainty matters less

### Term 2: ESP Uncertainty Contribution

$$k^2 \cdot \mu^2_{\text{CEC}} \cdot \sigma^2_{\text{ESP}}$$

- This term represents how uncertainty in ESP propagates to GR  
- It's weighted by the squared CEC mean
- **If CEC is high**, ESP uncertainty has a large impact
- **If CEC is low**, ESP uncertainty matters less

### Term 3: Cross-Covariance Contribution

$$2 k^2 \cdot (\mu_{\text{ESP}} - \text{ESP}_{\text{ref}}) \cdot \mu_{\text{CEC}} \cdot \text{Cov}[\text{CEC}, \text{ESP}]$$

- This term accounts for the correlation between ESP and CEC
- **If positive covariance**: Increases overall uncertainty (variables move together)
- **If negative covariance**: Decreases overall uncertainty (variables cancel out)
- **If zero covariance** (independence): This term disappears
- The factor of 2 comes from the symmetry of the mixed partial derivatives

---

## Cross-Covariance in Practice

### Converting Correlation to Covariance

As described earlier, we estimate a global correlation coefficient ρ from the training data, then convert it to location-specific covariance:

$$\text{Cov}[\text{ESP}(x), \text{CEC}(x)] = \rho \cdot \sigma_{\text{ESP}}(x) \cdot \sigma_{\text{CEC}}(x)$$

where σ_ESP(x) and σ_CEC(x) come from the GPR predictive standard deviations at location x.

### Spatial Variation of Cross-Covariance

Even though ρ is constant, the **absolute covariance varies spatially**:

- **Near training points**: Both σ_ESP(x) and σ_CEC(x) are small → small absolute covariance
- **Far from training points**: Both uncertainties are large → large absolute covariance
- **Mixed regions**: One variable certain, other uncertain → medium covariance

This makes intuitive sense: where we're uncertain about both variables, their correlation contributes more to the overall GR uncertainty.

### Sign of Correlation

- **ρ > 0** (positive correlation): High ESP tends to occur with high CEC
  - Common in many soils where both are influenced by clay content
  - **Effect on GR uncertainty**: Increases variance (both errors tend to align)
  
- **ρ < 0** (negative correlation): High ESP tends to occur with low CEC
  - Less common, but possible in some soil types
  - **Effect on GR uncertainty**: Decreases variance (errors tend to cancel)
  
- **ρ ≈ 0** (no correlation): ESP and CEC vary independently
  - Simplest case: third term in variance formula vanishes
  - **Effect on GR uncertainty**: No cross-correlation contribution

---

## Assumptions and Limitations

### Assumptions

1. **Local linearity**: The function is approximately linear near the means
2. **Small uncertainties**: Input uncertainties are small relative to the means
3. **Joint normality** (for rigorous justification): ESP and CEC follow a joint normal distribution

### When the Delta Method Works Well

✓ Input uncertainties are < 20-30% of the mean values  
✓ The function is smooth and continuously differentiable  
✓ You're interested in the mean and variance (not full distribution)  
✓ Computational efficiency is important (large spatial datasets)

### When to Consider Monte Carlo Instead

✗ Very large uncertainties (>50% of means)  
✗ Highly nonlinear relationships in the region of interest  
✗ Need for full probability distributions, not just moments  
✗ Non-Gaussian input distributions with heavy tails



## Implementation Notes

### Numerical Stability

1. **Check for negative variances**: Due to numerical precision, the variance formula might occasionally produce small negative values. Set these to zero:
   ```python
   variance = np.maximum(variance, 0)
   ```

2. **Regularization for covariance matrices**: When using Monte Carlo, ensure covariance matrices are positive semi-definite:
   ```python
   if np.any(np.linalg.eigvalsh(cov_matrix) < 0):
       cov_matrix += np.eye(2) * 1e-6
   ```

### Computational Efficiency

The Delta method is **highly efficient** for large spatial datasets:
- **Time complexity**: O(n) where n is the number of pixels
- **Memory**: Only stores means, variances, and covariance at each location
- **Parallelizable**: Each pixel can be computed independently

Compare to Monte Carlo:
- **Time complexity**: O(n × m) where m is the number of samples (typically 1000-10000)
- **Memory**: Must store all samples or compute statistics on-the-fly
- **Still parallelizable** but much slower

---

## Complete Post-Hoc Fusion Workflow

### Step-by-Step Process

#### Stage 1: Separate GPR Modeling

**1a. Train ESP GPR Model**
- Input: Training points {(x_i, ESP_i)} for i = 1, ..., n_ESP
- Output: Trained GPR model with optimized hyperparameters
- Can use different kernel (e.g., Matérn, RBF) best suited for ESP

**1b. Train CEC GPR Model**  
- Input: Training points {(x_j, CEC_j)} for j = 1, ..., n_CEC
- Output: Trained GPR model with optimized hyperparameters
- Can use different kernel and even different training locations
- Note: n_ESP and n_CEC don't need to be equal

**1c. Estimate Cross-Correlation**
- Find training points where both ESP and CEC were measured
- Compute Pearson correlation: ρ
- Store as global parameter

#### Stage 2: Generate Predictions

**2a. Predict ESP at Target Locations**
- Input: Prediction locations {x*₁, x*₂, ..., x*_m}
- Output: μ_ESP(x*), σ²_ESP(x*) at each location
- These are standard GPR predictions

**2b. Predict CEC at Target Locations**
- Input: Same prediction locations {x*₁, x*₂, ..., x*_m}  
- Output: μ_CEC(x*), σ²_CEC(x*) at each location
- GPR predictions independent of ESP predictions

#### Stage 3: Post-Hoc Fusion (Delta Method)

**3a. Compute Local Covariance**
- At each location x*: Cov[ESP(x*), CEC(x*)] = ρ · σ_ESP(x*) · σ_CEC(x*)

**3b. Apply Delta Method**
- Mean: μ_GR(x*) = k · μ_CEC(x*) · (μ_ESP(x*) - ESP_ref)
- Variance: Use full three-term formula with local covariance
- Standard deviation: σ_GR(x*) = √Var[GR(x*)]

**3c. Generate Output Maps**
- GR mean map (Mg/ha)
- GR uncertainty map (Mg/ha)
- Optional: confidence intervals, probability maps, etc.

### Advantages of This Workflow

1. **Modularity**: Each GPR model can be developed and validated independently
2. **Flexibility**: Different kernels, hyperparameters, and training data for each variable
3. **Efficiency**: Avoid expensive joint GP computations
4. **Scalability**: Can use sparse GP approximations independently for each variable
5. **Robustness**: If one GPR model fails or needs retraining, don't need to retrain both
6. **Interpretability**: Clear separation of model uncertainty and correlation effects

### Computational Complexity

**Joint Multivariate GP:**
- Training: O((2n)³) = O(8n³)
- Prediction: O((2n)²m) for m prediction points
- Memory: O((2n)²)

**Post-Hoc Fusion (Separate GPs + Delta Method):**
- Training: O(n³_ESP) + O(n³_CEC) ≈ O(2n³) if n_ESP ≈ n_CEC ≈ n
- Prediction: O(n²_ESP·m) + O(n²_CEC·m) + O(m) ≈ O(2n²m)
- Memory: O(n²_ESP) + O(n²_CEC) + O(m)
- **Speedup factor**: ~4x for training, ~2x for prediction when n_ESP = n_CEC

---

## Validation and Diagnostics

### How to Validate the Post-Hoc Fusion

1. **Check GPR Model Quality (separately)**
   - Cross-validation for ESP model
   - Cross-validation for CEC model
   - Ensure predictive distributions are well-calibrated

2. **Verify Correlation Estimate**
   - Plot ESP vs CEC at training locations
   - Check if correlation is stable across space
   - Consider computing local correlations if global assumption seems poor

3. **Compare with Monte Carlo**
   - Run Monte Carlo sampling for a subset of locations
   - Compare mean and variance from Delta method vs Monte Carlo
   - Large differences indicate strong nonlinearity or poor assumptions

4. **Reality Checks**
   - Do predicted GR values make agronomic sense?
   - Are uncertainties reasonable given data density?
   - Do high-uncertainty regions correspond to areas far from training data?

### When to Be Cautious

⚠️ **Very high correlation** (|ρ| > 0.8): Consider joint modeling or verify stability  
⚠️ **Spatially varying correlation**: Global ρ may be inadequate  
⚠️ **High nonlinearity**: ESP near ESP_ref makes function highly nonlinear  
⚠️ **Large uncertainties**: σ/μ > 0.5 suggests Delta method may be inaccurate  
⚠️ **Extrapolation**: Predictions far outside training range are unreliable

---

## Validating the Constant Correlation Assumption: Cross-Correlogram Analysis

### The Critical Assumption

The post-hoc probabilistic fusion approach relies on a key assumption: **the cross-correlation coefficient ρ between ESP and CEC is approximately constant across spatial distances**. This means we can use a single global value of ρ estimated from training data, rather than modeling a full spatially-varying cross-covariance function.

### What is a Cross-Correlogram?

A **cross-correlogram** (also called spatial cross-correlation function) plots the correlation between two variables as a function of spatial separation distance h:

$$\rho(h) = \frac{\text{Cov}[\text{ESP}(x), \text{CEC}(x+h)]}{\sigma_{\text{ESP}} \cdot \sigma_{\text{CEC}}}$$

where:
- h is the spatial lag distance (separation between locations)
- ESP(x) is the ESP value at location x
- CEC(x+h) is the CEC value at location x+h (h units away)

**Key interpretation:**
- **ρ(0)**: Correlation at lag-0 (same location) - this is what the Delta method uses
- **ρ(h) for h > 0**: Correlation between values at different locations separated by distance h
- **Flat ρ(h)**: Indicates constant correlation structure → post-hoc fusion is appropriate
- **Decaying ρ(h)**: Indicates spatially-varying correlation → joint multivariate GP recommended

### Why This Validation Matters

**If ρ(h) is approximately constant:**
- ✓ Post-hoc fusion with global ρ is valid
- ✓ Computational efficiency maintained (4x faster than joint GP)
- ✓ Simpler implementation using separate GPR models
- ✓ Uncertainty estimates are reliable

**If ρ(h) varies significantly with distance:**
- ✗ Post-hoc fusion may underestimate uncertainty
- ✗ Full cross-covariance kernel K_cross(x, x') is needed
- ✗ Requires joint multivariate GP modeling
- ✗ Higher computational cost but more accurate

### Computational Method

The cross-correlogram is computed by:

1. **Extract all valid pairs**: For locations i and j with valid ESP and CEC data
2. **Compute distances**: d_ij = ||x_i - x_j|| for all pairs
3. **Center the data**:
   - ESP_centered = ESP - mean(ESP)
   - CEC_centered = CEC - mean(CEC)
4. **Compute cross-products**: CP_ij = ESP_centered(x_i) × CEC_centered(x_j)
5. **Bin by distance**: Group pairs into distance bins (e.g., 20 bins from 0 to max_dist)
6. **Average per bin**: For bin k, compute mean cross-product
7. **Normalize**: Divide by σ_ESP × σ_CEC to get correlation

**Memory optimization**: For large datasets (>10,000 points), random subsampling is used since computing all O(n²) pairs is memory-intensive.

### Interpretation Guidelines

**Constant correlation (post-hoc fusion OK):**
- Standard deviation of ρ(h) < 0.05
- No significant trend with distance (p > 0.05)
- Visual plot shows approximately horizontal line
- Example: ρ(h) = 0.3 ± 0.02 across all lags

**Varying correlation (consider joint GP):**
- Standard deviation of ρ(h) > 0.1
- Significant decay with distance (common pattern)
- Range [min ρ(h), max ρ(h)] > 0.2
- Example: ρ(0) = 0.6, ρ(1000m) = 0.2

### Practical Recommendations

**Always run cross-correlogram analysis before using post-hoc fusion:**
1. Compute ρ(h) using `cross_correlogram.py`
2. Visually inspect the plot for trends
3. Check if std(ρ(h)) < 0.05 and range < 0.2
4. If validation passes → proceed with post-hoc fusion
5. If validation fails → consider joint multivariate GP

**Trade-offs:**
- Post-hoc fusion: ~1 second for 76k pixels, simple, ρ constant assumption
- Joint GP: ~60 seconds for 76k pixels, complex, learns ρ(h) from data

For operational mapping where ρ(h) is relatively constant, the post-hoc approach provides an excellent balance of accuracy and efficiency.

---

## Summary

The **post-hoc probabilistic fusion** approach for gypsum requirement mapping combines:

### Two-Stage Workflow

**Stage 1: Independent GPR Modeling**
- Fit separate Gaussian Process models for ESP and CEC
- Each model produces predictive means and variances at target locations
- Estimate global correlation ρ from training data where both variables were measured

**Stage 2: Delta Method Fusion**
- Mean GR: Evaluate function at predictive means from GPR
- Variance GR: Use Delta method with three terms:
  1. ESP uncertainty contribution: $k^2 \mu^2_{\text{CEC}} \sigma^2_{\text{ESP}}$
  2. CEC uncertainty contribution: $k^2 (\mu_{\text{ESP}} - \text{ESP}_{\text{ref}})^2 \sigma^2_{\text{CEC}}$
  3. Cross-correlation contribution: $2k^2 (\mu_{\text{ESP}} - \text{ESP}_{\text{ref}}) \mu_{\text{CEC}} \cdot \rho \sigma_{\text{ESP}} \sigma_{\text{CEC}}$

### When to Use This Method

**Best for:**
- Large spatial datasets (thousands to millions of prediction points)
- ESP and CEC measured at overlapping or nearby locations
- Moderate uncertainty levels (coefficient of variation < 30-50%)
- Need for computational efficiency

**Consider alternatives (Joint GP or Monte Carlo) when:**
- Very high correlation (|ρ| > 0.8) between variables
- Correlation varies strongly across space
- Very large uncertainties (coefficient of variation > 50%)
- Small datasets where computational cost isn't limiting
