"""
Improved Gypsum Requirement Calculation with Probabilistic Fusion

This module implements gypsum requirement calculation that properly accounts for
the joint distribution of ESP and CEC, including their cross-covariance structure.

Two methods are provided:
1. Analytical Approximation (Delta Method) - fast, first-order Taylor approximation
2. Monte Carlo Sampling - exact, but computationally intensive

Based on: GR_i = k × CEC_i × (ESP_i - ESP_ref)
where k = 0.086 * F * Ds * ρb
Note: 0.086 is for CEC in cmol_c/kg (0.0086 would be for mmol_c/kg)
"""

import numpy as np
import rasterio
from pathlib import Path
from scipy.stats import multivariate_normal
import warnings


def calculate_gypsum_probabilistic(
    esp_mean,
    esp_variance,
    cec_mean,
    cec_variance,
    esp_cec_covariance,
    esp_final=5.0,
    efficiency_factor=1.2,
    soil_depth=0.15,
    bulk_density=1.2,
    method='analytical'
):
    """
    Calculate gypsum requirement with probabilistic fusion accounting for cross-covariance.
    
    Parameters
    ----------
    esp_mean : numpy.ndarray
        Mean ESP values (%)
        
    esp_variance : numpy.ndarray
        ESP variance values (%)²
        
    cec_mean : numpy.ndarray
        Mean CEC values (cmol_c/kg)

    cec_variance : numpy.ndarray
        CEC variance values (cmol_c/kg)²
        
    esp_cec_covariance : numpy.ndarray or float
        Cross-covariance between ESP and CEC at each location
        If float (correlation coefficient), will be converted to covariance
        
    esp_final : float, optional
        Target final ESP (%), default = 5.0
        
    efficiency_factor : float, optional
        Ca-Na exchange efficiency factor, default = 1.2
        
    soil_depth : float, optional
        Soil depth (m), default = 0.15
        
    bulk_density : float, optional
        Bulk density (Mg/m³), default = 1.2
        
    method : str, optional
        'analytical' for Delta method or 'monte_carlo' for sampling
        Default = 'analytical'
        
    Returns
    -------
    gr_mean : numpy.ndarray
        Mean gypsum requirement (Mg/ha)
        
    gr_variance : numpy.ndarray
        Variance of gypsum requirement (Mg/ha)²
        
    gr_std : numpy.ndarray
        Standard deviation of gypsum requirement (Mg/ha)
        
    Notes
    -----
    Analytical approximation uses first-order Taylor expansion (Delta method).
    Monte Carlo method samples from the joint distribution (slower but exact).
    """
    
    # Calculate combined constant k = 0.086 * F * Ds * ρb
    # Note: 0.086 is calibrated for CEC in cmol_c/kg
    k = 0.086 * efficiency_factor * soil_depth * bulk_density
    
    # ESP difference from reference
    esp_diff = esp_mean - esp_final
    
    # Handle scalar correlation coefficient
    if isinstance(esp_cec_covariance, (int, float)):
        # Convert correlation to covariance
        esp_std = np.sqrt(esp_variance)
        cec_std = np.sqrt(cec_variance)
        esp_cec_covariance = esp_cec_covariance * esp_std * cec_std
    
    if method == 'analytical':
        return _calculate_analytical(
            k, esp_mean, esp_variance, cec_mean, cec_variance, 
            esp_cec_covariance, esp_final, esp_diff
        )
    elif method == 'monte_carlo':
        return _calculate_monte_carlo(
            k, esp_mean, esp_variance, cec_mean, cec_variance,
            esp_cec_covariance, esp_final, n_samples=1000
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'analytical' or 'monte_carlo'")


def _calculate_analytical(k, esp_mean, esp_variance, cec_mean, cec_variance, 
                          esp_cec_covariance, esp_final, esp_diff):
    """
    Analytical approximation using Delta method (first-order Taylor expansion).
    
    GR_mean ≈ k × μ_CEC × (μ_ESP - ESP_ref)
    
    Var[GR] ≈ k² × [(μ_ESP - ESP_ref)² × σ²_CEC + μ²_CEC × σ²_ESP + 
                     2(μ_ESP - ESP_ref) × μ_CEC × Cov[CEC, ESP]]
    """
    
    # Mean gypsum requirement
    gr_mean = k * cec_mean * esp_diff
    
    # Variance using Delta method
    # Partial derivatives evaluated at means:
    # ∂GR/∂CEC = k(ESP - ESP_ref)
    # ∂GR/∂ESP = k × CEC
    
    gr_variance = k**2 * (
        esp_diff**2 * cec_variance +
        cec_mean**2 * esp_variance +
        2 * esp_diff * cec_mean * esp_cec_covariance
    )
    
    # Handle negative variances (can occur due to approximation)
    gr_variance = np.maximum(gr_variance, 0)
    
    gr_std = np.sqrt(gr_variance)
    
    return gr_mean, gr_variance, gr_std


def _calculate_monte_carlo(k, esp_mean, esp_variance, cec_mean, cec_variance,
                           esp_cec_covariance, esp_final, n_samples=1000):
    """
    Monte Carlo approximation by sampling from joint distribution.

    For each location, samples from bivariate normal distribution and
    computes GR = k × CEC × (ESP - ESP_ref) for each sample.
    """

    n_locations = len(esp_mean)
    gr_samples = np.zeros((n_samples, n_locations))

    # Pre-filter invalid locations
    valid_mask = (
        np.isfinite(esp_mean) &
        np.isfinite(esp_variance) &
        np.isfinite(cec_mean) &
        np.isfinite(cec_variance) &
        np.isfinite(esp_cec_covariance) &
        (esp_variance > 0) &
        (cec_variance > 0)
    )

    n_valid = valid_mask.sum()
    n_invalid = (~valid_mask).sum()

    if n_invalid > 0:
        print(f"  Skipping {n_invalid} invalid locations, processing {n_valid} valid locations")

    for i in range(n_locations):
        # Skip invalid locations
        if not valid_mask[i]:
            gr_samples[:, i] = np.nan
            continue

        # Construct 2x2 covariance matrix for this location
        cov_matrix = np.array([
            [cec_variance[i], esp_cec_covariance[i]],
            [esp_cec_covariance[i], esp_variance[i]]
        ])

        # Check matrix validity
        eigvals = np.linalg.eigvalsh(cov_matrix)
        if np.any(eigvals < -1e-10):  # Allow small numerical errors
            # Add regularization to diagonal
            min_eigval = np.min(eigvals)
            cov_matrix += np.eye(2) * (abs(min_eigval) + 1e-8)

        # Sample from joint distribution
        try:
            samples = multivariate_normal.rvs(
                mean=[cec_mean[i], esp_mean[i]],
                cov=cov_matrix,
                size=n_samples
            )

            if n_samples == 1:
                samples = samples.reshape(1, -1)

            cec_samples = samples[:, 0]
            esp_samples = samples[:, 1]

            # Calculate GR for each sample
            gr_samples[:, i] = k * cec_samples * (esp_samples - esp_final)

        except Exception as e:
            warnings.warn(f"Sampling failed at location {i}: {e}. Using analytical approximation.")
            # Fallback to analytical
            gr_mean_i, gr_var_i, _ = _calculate_analytical(
                k,
                np.array([esp_mean[i]]),
                np.array([esp_variance[i]]),
                np.array([cec_mean[i]]),
                np.array([cec_variance[i]]),
                np.array([esp_cec_covariance[i]]),
                esp_final,
                np.array([esp_mean[i] - esp_final])
            )
            gr_samples[:, i] = gr_mean_i[0]

    # Compute statistics from samples (ignoring NaN locations)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Degrees of freedom.*')
        gr_mean = np.nanmean(gr_samples, axis=0)
        gr_variance = np.nanvar(gr_samples, axis=0, ddof=1)
        gr_std = np.nanstd(gr_samples, axis=0, ddof=1)

    return gr_mean, gr_variance, gr_std


def estimate_cross_covariance_from_data(esp_data, cec_data, esp_unc, cec_unc, method='correlation'):
    """
    Estimate cross-covariance between ESP and CEC from observed data.
    
    Parameters
    ----------
    esp_data : numpy.ndarray
        ESP measurements
        
    cec_data : numpy.ndarray
        CEC measurements
        
    esp_unc : numpy.ndarray
        ESP uncertainties (standard deviations, %)

    cec_unc : numpy.ndarray
        CEC uncertainties (standard deviations, cmol_c/kg)
        
    method : str, optional
        'correlation' - assumes constant correlation across space
        'local' - computes local covariance (not yet implemented)
        
    Returns
    -------
    cross_covariance : numpy.ndarray or float
        Cross-covariance at each location (or scalar if method='correlation')
    """
    
    if method == 'correlation':
        # Compute global correlation coefficient
        valid_mask = np.isfinite(esp_data) & np.isfinite(cec_data)
        
        if np.sum(valid_mask) < 2:
            warnings.warn("Insufficient valid data for correlation. Assuming independence (rho=0)")
            return 0.0
        
        rho = np.corrcoef(esp_data[valid_mask], cec_data[valid_mask])[0, 1]
        
        if not np.isfinite(rho):
            warnings.warn("Invalid correlation computed. Assuming independence (rho=0)")
            rho = 0.0
        
        print(f"Estimated correlation coefficient: {rho:.3f}")
        
        # Return as scalar - will be converted to covariance in main function
        return rho
    
    else:
        raise NotImplementedError(f"Method '{method}' not yet implemented")


def calculate_gypsum_from_geotiff_probabilistic(
    esp_initial_tif,
    esp_initial_uncertainty_tif,
    cec_tif,
    cec_uncertainty_tif,
    output_dir,
    output_prefix="gypsum_prob",
    esp_final=5.0,
    efficiency_factor=1.2,
    soil_depth=0.15,
    bulk_density=1.2,
    method='analytical',
    correlation_coefficient=None,
    nodata_value=-9999
):
    """
    Calculate gypsum requirement from GeoTIFF files using probabilistic fusion.
    
    Parameters
    ----------
    esp_initial_tif : str or Path
        Path to ESP mean values GeoTIFF
        
    esp_initial_uncertainty_tif : str or Path
        Path to ESP uncertainty (std dev) GeoTIFF (%)

    cec_tif : str or Path
        Path to CEC mean values GeoTIFF (cmol_c/kg)

    cec_uncertainty_tif : str or Path
        Path to CEC uncertainty (std dev) GeoTIFF (cmol_c/kg)
        
    output_dir : str or Path
        Output directory for results
        
    output_prefix : str, optional
        Prefix for output files, default = "gypsum_prob"
        
    esp_final : float, optional
        Target ESP (%), default = 5.0
        
    efficiency_factor : float, optional
        Exchange efficiency, default = 1.2
        
    soil_depth : float, optional
        Soil depth (m), default = 0.15
        
    bulk_density : float, optional
        Bulk density (Mg/m³), default = 1.2
        
    method : str, optional
        'analytical' or 'monte_carlo', default = 'analytical'
        
    correlation_coefficient : float, optional
        ESP-CEC correlation. If None, estimated from data
        
    nodata_value : float, optional
        NoData value for outputs, default = -9999
        
    Returns
    -------
    output_files : dict
        Dictionary of output file paths
    """
    
    # Convert paths
    esp_initial_tif = Path(esp_initial_tif)
    esp_initial_uncertainty_tif = Path(esp_initial_uncertainty_tif)
    cec_tif = Path(cec_tif)
    cec_uncertainty_tif = Path(cec_uncertainty_tif)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PROBABILISTIC GYPSUM REQUIREMENT CALCULATION")
    print("="*70)
    print(f"Method: {method.upper()}")
    print(f"Reading input files...")
    
    # Read all inputs
    with rasterio.open(esp_initial_tif) as src:
        esp_mean = src.read(1)
        profile = src.profile.copy()
        shape = esp_mean.shape
        
    with rasterio.open(esp_initial_uncertainty_tif) as src:
        esp_std = src.read(1)
        
    with rasterio.open(cec_tif) as src:
        cec_mean = src.read(1)
        
    with rasterio.open(cec_uncertainty_tif) as src:
        cec_std = src.read(1)
    
    print(f"Raster shape: {shape}")
    
    # Convert uncertainties to variances
    esp_variance = esp_std ** 2
    cec_variance = cec_std ** 2
    
    # Handle NoData
    nodata_input = profile.get('nodata', nodata_value)
    mask = (
        (esp_mean == nodata_input) | 
        (cec_mean == nodata_input) |
        ~np.isfinite(esp_mean) |
        ~np.isfinite(cec_mean)
    )
    
    # Estimate or use provided correlation coefficient
    if correlation_coefficient is None:
        print("\nEstimating ESP-CEC correlation from data...")
        esp_flat = esp_mean[~mask]
        cec_flat = cec_mean[~mask]
        correlation_coefficient = estimate_cross_covariance_from_data(
            esp_flat, cec_flat, 
            esp_std[~mask], cec_std[~mask],
            method='correlation'
        )
    else:
        print(f"\nUsing provided correlation coefficient: {correlation_coefficient:.3f}")
    
    # Calculate gypsum requirement
    print(f"\nCalculating gypsum requirement...")
    print(f"  ESP_final = {esp_final}%")
    print(f"  k = 0.086 × {efficiency_factor} × {soil_depth} × {bulk_density} = {0.086*efficiency_factor*soil_depth*bulk_density:.6f}")
    
    gr_mean, gr_variance, gr_std = calculate_gypsum_probabilistic(
        esp_mean.flatten(),
        esp_variance.flatten(),
        cec_mean.flatten(),
        cec_variance.flatten(),
        correlation_coefficient,
        esp_final=esp_final,
        efficiency_factor=efficiency_factor,
        soil_depth=soil_depth,
        bulk_density=bulk_density,
        method=method
    )
    
    # Reshape to original shape
    gr_mean = gr_mean.reshape(shape)
    gr_variance = gr_variance.reshape(shape)
    gr_std = gr_std.reshape(shape)
    
    # Apply mask and set negative values to zero
    gr_mean = np.where(mask, nodata_value, gr_mean)
    gr_std = np.where(mask, nodata_value, gr_std)
    
    # Flag where gypsum not needed (ESP already below target)
    no_gypsum_flag = (esp_mean <= esp_final) | mask
    gr_mean = np.where(no_gypsum_flag & ~mask, 0.0, gr_mean)
    gr_std = np.where(no_gypsum_flag & ~mask, 0.0, gr_std)
    
    # Prepare output files
    output_profile = profile.copy()
    output_profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw',
        nodata=nodata_value
    )
    
    output_files = {
        'gypsum_mean': output_dir / f"{output_prefix}_mean_Mg_ha.tif",
        'gypsum_std': output_dir / f"{output_prefix}_std_Mg_ha.tif",
        'no_gypsum_flag': output_dir / f"{output_prefix}_no_application_flag.tif"
    }
    
    # Write outputs
    print(f"\nWriting outputs to: {output_dir}")
    
    with rasterio.open(output_files['gypsum_mean'], 'w', **output_profile) as dst:
        dst.write(gr_mean.astype(rasterio.float32), 1)
        dst.set_band_description(1, f"Gypsum Requirement Mean (Mg/ha) - {method}")
    print(f"  ✓ {output_files['gypsum_mean'].name}")
    
    with rasterio.open(output_files['gypsum_std'], 'w', **output_profile) as dst:
        dst.write(gr_std.astype(rasterio.float32), 1)
        dst.set_band_description(1, f"Gypsum Requirement Std Dev (Mg/ha) - {method}")
    print(f"  ✓ {output_files['gypsum_std'].name}")
    
    flag_profile = output_profile.copy()
    flag_profile.update(dtype=rasterio.uint8, nodata=255)
    with rasterio.open(output_files['no_gypsum_flag'], 'w', **flag_profile) as dst:
        dst.write(no_gypsum_flag.astype(rasterio.uint8), 1)
        dst.set_band_description(1, "No Gypsum Needed Flag")
    print(f"  ✓ {output_files['no_gypsum_flag'].name}")
    
    # Print statistics
    valid_gr = gr_mean[~mask & ~no_gypsum_flag]
    valid_std = gr_std[~mask & ~no_gypsum_flag]
    
    if len(valid_gr) > 0:
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(f"Total pixels: {(~mask).sum()}")
        print(f"Pixels requiring gypsum: {len(valid_gr)}")
        print(f"Pixels below threshold: {np.sum(~mask & no_gypsum_flag)}")
        print(f"\nGypsum Requirement (Mg/ha):")
        print(f"  Mean ± Std:  {np.mean(valid_gr):.2f} ± {np.mean(valid_std):.2f}")
        print(f"  Median:      {np.median(valid_gr):.2f}")
        print(f"  Range:       [{np.min(valid_gr):.2f}, {np.max(valid_gr):.2f}]")
        print(f"  P05-P95:     [{np.percentile(valid_gr, 5):.2f}, {np.percentile(valid_gr, 95):.2f}]")
        print(f"\nUncertainty (Std Dev):")
        print(f"  Mean:        {np.mean(valid_std):.2f}")
        print(f"  Median:      {np.median(valid_std):.2f}")
        print(f"  Range:       [{np.min(valid_std):.2f}, {np.max(valid_std):.2f}]")
        print("="*70)
    
    print("\n✓ Processing complete!")
    
    return output_files


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python gypsum_probabilistic.py <esp.tif> <esp_unc.tif> <cec.tif> <cec_unc.tif> [output_dir] [method]")
        print("\nMethods: analytical (default), monte_carlo")
        print("\nExample:")
        print("  python gypsum_probabilistic.py esp.tif esp_unc.tif cec.tif cec_unc.tif results analytical")
        sys.exit(1)
    
    output_dir = sys.argv[5] if len(sys.argv) > 5 else "gypsum_probabilistic_output"
    method = sys.argv[6] if len(sys.argv) > 6 else "analytical"
    
    output_files = calculate_gypsum_from_geotiff_probabilistic(
        esp_initial_tif=sys.argv[1],
        esp_initial_uncertainty_tif=sys.argv[2],
        cec_tif=sys.argv[3],
        cec_uncertainty_tif=sys.argv[4],
        output_dir=output_dir,
        method=method
    )
    
    print("\nOutput files:")
    for key, path in output_files.items():
        print(f"  {key}: {path}")
