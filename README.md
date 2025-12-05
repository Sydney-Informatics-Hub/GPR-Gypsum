# Gypsum Requirement Calculation via Probabilistic Fusion

**Author**: Sebastian Haan

A suite of Python tools for calculating soil gypsum requirements through probabilistic fusion of geospatial predictions of ESP (Exchangeable Sodium Percentage) and CEC (Cation Exchange Capacity), with uncertainty propagation and cross-correlation handling for spatially-distributed soil property data.

## Overview

These scripts implement the gypsum requirement equation from Oster et al. (Agricultural Management of Sodic Soils):

**GR = 0.086 × F × Ds × ρb × CEC × (ESP - ESP_ref)**

where:
- GR = Gypsum requirement (Mg/ha)
- F = Ca-Na exchange efficiency factor (default: 1.2)
- Ds = Soil depth in meters (default: 0.15 m = 15 cm)
- ρb = Bulk density (Mg/m³) (default: 1.2)
- CEC = Cation exchange capacity (cmol_c/kg)
- ESP = Exchangeable sodium percentage (%)
- ESP_ref = Target ESP threshold (default: 5%)

**Note**: The conversion factor 0.086 is calibrated for CEC in **cmol_c/kg**. If your CEC data is in mmol_c/kg, use 0.0086 instead.

---

## Related Projects

This repository provides **post-hoc probabilistic fusion** tools for combining predictions from separate spatial models. It is designed as an add-on to:

**[AgReFed-ML](https://github.com/Sydney-Informatics-Hub/AgReFed-ML)** - A comprehensive suite for modelling and predicting agricultural systems and their uncertainties using Gaussian Process Regression and other machine learning methods.

**Workflow**:
1. Use AgReFed-ML to generate independent GPR predictions for ESP and CEC (with uncertainties)
2. Use this repository to perform post-hoc probabilistic fusion and calculate gypsum requirements accounting for cross-correlation

---

## Scripts

### 1. `cross_correlogram.py`
**Purpose**: Analyse spatial cross-correlation between ESP and CEC

**Use when**: You need to understand the correlation structure between ESP and CEC in your field, which is essential for the probabilistic approach.

**Inputs**:
- ESP GeoTIFF (mean values)
- CEC GeoTIFF (mean values)

**Outputs**:
- Cross-correlogram plot (correlation vs distance)
- CSV file with lag distances and correlations
- JSON metadata file
- Estimated global correlation coefficient ρ

**Example**:
```bash
python cross_correlogram.py ESP.tif CEC.tif results
```

```python
from cross_correlogram import compute_cross_correlogram_from_geotiff

results = compute_cross_correlogram_from_geotiff(
    esp_tif='esp_mean.tif',
    cec_tif='cec_mean.tif',
    output_dir='correlation_analysis',
    n_bins=20
)
print(f"Global correlation: {results['esp_mean']:.3f}")
```

---

### 2. `gypsum_requirement.py`
**Purpose**: Core calculation functions (used by gypsum_geotiff.py)

**Contains**:
- `calculate_gypsum_requirement()`: Array-based calculation
- `summarize_gypsum_application()`: Summary statistics helper

**Note**: This is a **dependency** for `gypsum_geotiff.py`. Most users won't call this directly unless working with arrays rather than GeoTIFFs.

---

### 3. `gypsum_geotiff.py` ⚠️ BASIC APPROACH
**Purpose**: Calculate gypsum requirement from GeoTIFFs **without cross-correlation**

**Use when**: 
- Quick analysis needed
- ESP and CEC are approximately independent (|ρ| < 0.3)
- No GPR predictions available (direct measurements only)

**Limitations**: 
- ⚠️ Assumes ESP and CEC are **independent** (ignores correlation)
- ⚠️ May **underestimate or overestimate uncertainty** if variables are correlated
- Simpler uncertainty propagation

**Inputs**:
- 4 GeoTIFF files: ESP mean, ESP uncertainty, CEC mean, CEC uncertainty

**Outputs**:
- Gypsum requirement (Mg/ha)
- Gypsum uncertainty (Mg/ha)
- No-application flag (binary)

**Example**:
```bash
python gypsum_geotiff.py esp.tif esp_unc.tif cec.tif cec_unc.tif results
```

```python
from gypsum_geotiff import calculate_gypsum_from_geotiff

output_files = calculate_gypsum_from_geotiff(
    esp_initial_tif='esp_mean.tif',
    esp_initial_uncertainty_tif='esp_std.tif',
    cec_tif='cec_mean.tif',
    cec_uncertainty_tif='cec_std.tif',
    output_dir='basic_results'
)
```

---

### 4. `gypsum_probabilistic.py` ✅ RECOMMENDED APPROACH
**Purpose**: Calculate gypsum requirement with **proper cross-correlation handling**

**Use when**:
- ESP and CEC predictions from Gaussian Process Regression
- ESP and CEC are correlated (|ρ| > 0.3)
- Need accurate uncertainty quantification
- Post-hoc probabilistic fusion needed

**Key Features**:
- ✅ Accounts for ESP-CEC cross-correlation
- ✅ Two methods: analytical (fast) and Monte Carlo (exact)
- ✅ Automatically estimates correlation from data
- ✅ Proper Delta method uncertainty propagation

**Inputs**:
- 4 GeoTIFF files: ESP mean, ESP uncertainty (from GPR), CEC mean, CEC uncertainty (from GPR)
- Optional: pre-computed correlation coefficient ρ

**Outputs**:
- Gypsum requirement mean (Mg/ha)
- Gypsum requirement std dev (Mg/ha)
- No-application flag (binary)
- Summary statistics

**Example**:
```bash
# Analytical method (recommended, fast)
python gypsum_probabilistic.py esp.tif esp_unc.tif cec.tif cec_unc.tif results analytical

# Monte Carlo method (exact but slower)
python gypsum_probabilistic.py esp.tif esp_unc.tif cec.tif cec_unc.tif results monte_carlo
```

```python
from gypsum_probabilistic import calculate_gypsum_from_geotiff_probabilistic

# Let it estimate correlation from data
output_files = calculate_gypsum_from_geotiff_probabilistic(
    esp_initial_tif='esp_gpr_mean.tif',
    esp_initial_uncertainty_tif='esp_gpr_std.tif',
    cec_tif='cec_gpr_mean.tif',
    cec_uncertainty_tif='cec_gpr_std.tif',
    output_dir='probabilistic_results',
    method='analytical'  # or 'monte_carlo'
)

# Or provide known correlation
output_files = calculate_gypsum_from_geotiff_probabilistic(
    esp_initial_tif='esp_gpr_mean.tif',
    esp_initial_uncertainty_tif='esp_gpr_std.tif',
    cec_tif='cec_gpr_mean.tif',
    cec_uncertainty_tif='cec_gpr_std.tif',
    output_dir='probabilistic_results',
    correlation_coefficient=0.45,  # from cross_correlogram analysis
    method='analytical'
)
```

---

## Recommended Workflow

### Complete Analysis Pipeline

```bash
# Step 1: Analyze correlation structure (run once for your field)
python cross_correlogram.py esp_gpr_mean.tif cec_gpr_mean.tif correlation_analysis 20

# Step 2: Calculate gypsum requirement with proper uncertainty
python gypsum_probabilistic.py esp_gpr_mean.tif esp_gpr_std.tif cec_gpr_mean.tif cec_gpr_std.tif final_results analytical
```

### Decision Tree: Which Script to Use?

```
Do you have GPR predictions for ESP and CEC?
│
├─ YES → Are ESP and CEC correlated (|ρ| > 0.3)?
│         │
│         ├─ YES → Use gypsum_probabilistic.py ✅ (RECOMMENDED)
│         │        First run cross_correlogram.py to estimate ρ
│         │
│         └─ NO → Can use gypsum_geotiff.py (simpler, faster)
│
└─ NO → Do you have uncertainty estimates?
          │
          ├─ YES → Use gypsum_geotiff.py (basic approach)
          │
          └─ NO → Need to estimate uncertainties first
                   Consider using GPR or other spatial methods
```

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- rasterio
- scipy
- matplotlib
- pandas

### Optional: Virtual Environment

```bash
# Create virtual environment
python -m venv gypsum_env

# Activate (Linux/Mac)
source gypsum_env/bin/activate

# Activate (Windows)
gypsum_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Input File Requirements

### GeoTIFF Format

All input GeoTIFF files must:
- ✓ Have the **same dimensions** (rows × columns)
- ✓ Have the **same coordinate reference system** (CRS)
- ✓ Have the **same geotransform** (resolution and alignment)
- ✓ Use consistent NoData values

### Units

- **ESP**: Percentage (%) - typically 0-100
- **CEC**: cmol_c/kg (centimoles of charge per kilogram) or meq/100g (equivalent units)
- **Uncertainties**: Same units as the respective variable (std dev, not variance)

### Example: Creating Compatible Rasters

```python
import rasterio
import numpy as np

# Read reference raster
with rasterio.open('esp_mean.tif') as src:
    profile = src.profile
    esp_data = src.read(1)

# Create uncertainty raster with same profile
esp_std = esp_data * 0.15  # Example: 15% relative uncertainty

with rasterio.open('esp_std.tif', 'w', **profile) as dst:
    dst.write(esp_std, 1)
```

---

## Default Parameters

All scripts use these default values (based on Oster et al.):

| Parameter | Symbol | Default Value | Unit | Description |
|-----------|--------|---------------|------|-------------|
| Exchange efficiency | F | 1.2 | - | Ca-Na exchange factor |
| Soil depth | Ds | 0.15 | m | Treatment depth (15 cm) |
| Bulk density | ρb | 1.2 | Mg/m³ | Soil bulk density |
| Target ESP | ESP_ref | 5.0 | % | Below sodicity threshold (6%) |

### Modifying Defaults

```python
output_files = calculate_gypsum_from_geotiff_probabilistic(
    esp_initial_tif='esp.tif',
    esp_initial_uncertainty_tif='esp_unc.tif',
    cec_tif='cec.tif',
    cec_uncertainty_tif='cec_unc.tif',
    output_dir='results',
    # Custom parameters
    esp_final=6.0,          # Different target ESP
    efficiency_factor=1.3,   # Different F value
    soil_depth=0.20,        # 20 cm depth
    bulk_density=1.4        # Higher bulk density
)
```

---

## Output Files

### Gypsum Requirement Maps

**Basic approach** (`gypsum_geotiff.py`):
- `{prefix}_requirement_Mg_ha.tif` - Gypsum requirement values
- `{prefix}_uncertainty_Mg_ha.tif` - Uncertainty (std dev)
- `{prefix}_no_application_flag.tif` - Binary flag (1=no gypsum needed, 0=gypsum needed)

**Probabilistic approach** (`gypsum_probabilistic.py`):
- `{prefix}_mean_Mg_ha.tif` - Mean gypsum requirement
- `{prefix}_std_Mg_ha.tif` - Standard deviation
- `{prefix}_no_application_flag.tif` - Binary flag

### Cross-Correlogram Output

- `{prefix}_data.csv` - Lag distances, correlations, and pair counts
- `{prefix}_metadata.json` - Summary statistics and input info
- `{prefix}_plot.png` - Visualization of correlogram

---

## Interpreting Results

### Gypsum Requirement Values

- **Zero values**: ESP already below target threshold (no treatment needed)
- **Low values (0-5 Mg/ha)**: Minor sodicity, light treatment
- **Medium values (5-15 Mg/ha)**: Moderate sodicity, standard treatment
- **High values (>15 Mg/ha)**: Severe sodicity, heavy treatment required

### Uncertainty Values

- **Relative uncertainty** = (std dev / mean) × 100%
- **Low uncertainty (<20%)**: Confident predictions, close to measurements
- **Medium uncertainty (20-40%)**: Moderate confidence, typical for interpolation
- **High uncertainty (>40%)**: Low confidence, far from measurements, consider more sampling

### No-Application Flag

- **0**: Gypsum application recommended
- **1**: ESP already below threshold, no gypsum needed
- **255** (NoData): Invalid data at this location

---

## Troubleshooting

### Common Errors

**"Input GeoTIFFs do not have matching dimensions"**
- Solution: Resample all rasters to the same grid using GDAL or rasterio

**"Invalid correlation computed. Assuming independence (rho=0)"**
- Cause: Insufficient overlapping valid data points
- Solution: Check NoData values and ensure ESP/CEC have adequate overlap

**"Sampling failed at location X"**
- Cause: Negative variance or singular covariance matrix
- Solution: Check for very small or negative uncertainties, may indicate data issues

**"Memory error"**
- Cause: Very large rasters with Monte Carlo method
- Solution: Use analytical method instead, or process in tiles

### Validation Checks

```python
# Check correlation estimate
from cross_correlogram import compute_cross_correlogram_from_geotiff
results = compute_cross_correlogram_from_geotiff('esp.tif', 'cec.tif', 'check')

# Verify |ρ| is reasonable (typically < 0.9)
# Plot looks sensible (correlation decreases with distance)

# Compare methods
from gypsum_probabilistic import calculate_gypsum_from_geotiff_probabilistic

# Run both
out_analytical = calculate_gypsum_from_geotiff_probabilistic(
    ..., method='analytical', output_prefix='test_analytical'
)
out_mc = calculate_gypsum_from_geotiff_probabilistic(
    ..., method='monte_carlo', output_prefix='test_mc'
)

# Compare the results - should be similar if assumptions hold
```

---

## Performance Tips

### For Large Datasets (>10M pixels)

1. **Use analytical method** (not Monte Carlo) - is faster
2. **Process in tiles** if memory limited
3. **Use compression** in output GeoTIFFs (LZW compression enabled by default)
4. **Consider downsampling** for exploratory analysis


---

## Acknowledgments

Acknowledgments are an important way for us to demonstrate the value we bring to your research. Your research outcomes are vital for ongoing funding of the Sydney Informatics Hub.

If you make use of this software for your research project, please include the following acknowledgment:

"This research was supported by the Sydney Informatics Hub, a Core Research Facility of the University of Sydney."

---

## Documentation

### DESCRIPTION.md - Mathematical Foundation

**`DESCRIPTION.md`** provides comprehensive documentation of the **post-hoc probabilistic fusion** method used in `gypsum_probabilistic.py`:

**Contents**:
- Delta method mathematical derivation
- Post-hoc fusion workflow (separate GPR models → fused predictions)
- Variance propagation with three-term formula (ESP, CEC, and cross-correlation contributions)
- Cross-correlogram validation methods
- Computational complexity analysis
- Assumptions, limitations, and when to use alternatives

**Key sections**:
1. **Problem Statement**: GPR outputs and gypsum requirement equation
2. **Why Post-Hoc Fusion**: Advantages over joint multivariate GP modeling
3. **Delta Method Application**: Step-by-step variance calculation
4. **Cross-Covariance in Practice**: Converting correlation to location-specific covariance
5. **Validation Guidelines**: Cross-correlogram analysis and diagnostics

**When to read**:
- Understanding the theoretical foundation of `gypsum_probabilistic.py`
- Validating the constant correlation assumption
- Comparing analytical vs Monte Carlo approaches
- Troubleshooting uncertainty estimates

**PDF version**: `DESCRIPTION.pdf` (generated from markdown)

---

## Quick Reference Card

```bash
# STEP 1: Analyze correlation (run once)
python cross_correlogram.py esp.tif cec.tif correlation/ 20

# STEP 2: Calculate gypsum requirement
# Choose ONE of:

# Option A: Basic (no correlation, faster)
python gypsum_geotiff.py esp.tif esp_std.tif cec.tif cec_std.tif basic_output/

# Option B: Probabilistic (with correlation, recommended)
python gypsum_probabilistic.py esp.tif esp_std.tif cec.tif cec_std.tif prob_output/ analytical
```

**Default outputs**: `{prefix}_mean_Mg_ha.tif`, `{prefix}_std_Mg_ha.tif`, `{prefix}_no_application_flag.tif`

**Documentation**: See `DESCRIPTION.md` for mathematical foundations and detailed methodology

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
