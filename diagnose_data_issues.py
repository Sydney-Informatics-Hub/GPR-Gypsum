"""
Diagnostic script to identify data quality issues causing Monte Carlo sampling failures.
"""

import numpy as np
import rasterio
from pathlib import Path
import sys


def diagnose_geotiff_quality(esp_mean_tif, esp_std_tif, cec_mean_tif, cec_std_tif):
    """
    Diagnose data quality issues in input GeoTIFFs.
    """

    print("="*70)
    print("DATA QUALITY DIAGNOSTIC REPORT")
    print("="*70)

    # Read all files
    with rasterio.open(esp_mean_tif) as src:
        esp_mean = src.read(1)
        nodata = src.nodata

    with rasterio.open(esp_std_tif) as src:
        esp_std = src.read(1)

    with rasterio.open(cec_mean_tif) as src:
        cec_mean = src.read(1)

    with rasterio.open(cec_std_tif) as src:
        cec_std = src.read(1)

    print(f"\nRaster shape: {esp_mean.shape}")
    print(f"Total pixels: {esp_mean.size}")
    print(f"NoData value: {nodata}")

    # Create valid data mask
    if nodata is not None:
        mask = (
            (esp_mean == nodata) |
            (cec_mean == nodata) |
            (esp_std == nodata) |
            (cec_std == nodata)
        )
    else:
        mask = np.zeros_like(esp_mean, dtype=bool)

    # Add inf/NaN to mask
    mask = mask | ~np.isfinite(esp_mean) | ~np.isfinite(cec_mean) | \
           ~np.isfinite(esp_std) | ~np.isfinite(cec_std)

    n_valid = (~mask).sum()
    print(f"Valid pixels: {n_valid} ({100*n_valid/esp_mean.size:.1f}%)")

    # Check for inf/NaN values
    print("\n" + "="*70)
    print("INVALID VALUES (inf/NaN)")
    print("="*70)

    for name, arr in [('ESP mean', esp_mean), ('ESP std', esp_std),
                      ('CEC mean', cec_mean), ('CEC std', cec_std)]:
        n_inf = np.isinf(arr).sum()
        n_nan = np.isnan(arr).sum()
        print(f"{name:12s}: {n_inf:8d} inf, {n_nan:8d} NaN")

    # Check for negative or zero values in uncertainties
    print("\n" + "="*70)
    print("UNCERTAINTY ISSUES")
    print("="*70)

    esp_std_valid = esp_std[~mask]
    cec_std_valid = cec_std[~mask]

    n_esp_negative = (esp_std_valid < 0).sum()
    n_esp_zero = (esp_std_valid == 0).sum()
    n_esp_tiny = ((esp_std_valid > 0) & (esp_std_valid < 1e-10)).sum()

    n_cec_negative = (cec_std_valid < 0).sum()
    n_cec_zero = (cec_std_valid == 0).sum()
    n_cec_tiny = ((cec_std_valid > 0) & (cec_std_valid < 1e-10)).sum()

    print(f"ESP std: {n_esp_negative:8d} negative, {n_esp_zero:8d} zero, {n_esp_tiny:8d} very small (<1e-10)")
    print(f"CEC std: {n_cec_negative:8d} negative, {n_cec_zero:8d} zero, {n_cec_tiny:8d} very small (<1e-10)")

    # Check for extreme values
    print("\n" + "="*70)
    print("VALUE RANGES (valid pixels only)")
    print("="*70)

    for name, arr in [('ESP mean', esp_mean), ('ESP std', esp_std),
                      ('CEC mean', cec_mean), ('CEC std', cec_std)]:
        arr_valid = arr[~mask]
        if len(arr_valid) > 0:
            print(f"\n{name}:")
            print(f"  Min:    {np.min(arr_valid):12.6f}")
            print(f"  Max:    {np.max(arr_valid):12.6f}")
            print(f"  Mean:   {np.mean(arr_valid):12.6f}")
            print(f"  Median: {np.median(arr_valid):12.6f}")
            print(f"  P01:    {np.percentile(arr_valid, 1):12.6f}")
            print(f"  P99:    {np.percentile(arr_valid, 99):12.6f}")

    # Check relative uncertainty
    print("\n" + "="*70)
    print("RELATIVE UNCERTAINTY (std/mean)")
    print("="*70)

    esp_mean_valid = esp_mean[~mask]
    rel_unc_esp = esp_std_valid / np.abs(esp_mean_valid + 1e-10)
    rel_unc_cec = cec_std_valid / np.abs(cec_mean[~mask] + 1e-10)

    print(f"\nESP relative uncertainty:")
    print(f"  Mean:   {np.mean(rel_unc_esp):.3f}")
    print(f"  Median: {np.median(rel_unc_esp):.3f}")
    print(f"  P95:    {np.percentile(rel_unc_esp, 95):.3f}")
    print(f"  Max:    {np.max(rel_unc_esp):.3f}")
    print(f"  Pixels with >50% uncertainty: {(rel_unc_esp > 0.5).sum()}")
    print(f"  Pixels with >100% uncertainty: {(rel_unc_esp > 1.0).sum()}")

    print(f"\nCEC relative uncertainty:")
    print(f"  Mean:   {np.mean(rel_unc_cec):.3f}")
    print(f"  Median: {np.median(rel_unc_cec):.3f}")
    print(f"  P95:    {np.percentile(rel_unc_cec, 95):.3f}")
    print(f"  Max:    {np.max(rel_unc_cec):.3f}")
    print(f"  Pixels with >50% uncertainty: {(rel_unc_cec > 0.5).sum()}")
    print(f"  Pixels with >100% uncertainty: {(rel_unc_cec > 1.0).sum()}")

    # Check covariance matrix validity
    print("\n" + "="*70)
    print("COVARIANCE MATRIX ISSUES")
    print("="*70)

    # Estimate correlation
    if len(esp_mean_valid) > 1:
        rho = np.corrcoef(esp_mean_valid, cec_mean[~mask])[0, 1]
        print(f"\nEstimated correlation coefficient: {rho:.3f}")

        # Check for locations that would have invalid covariance matrices
        esp_var = esp_std**2
        cec_var = cec_std**2
        cov = rho * esp_std * cec_std

        # Determinant = var_esp * var_cec - cov^2
        # Must be >= 0 for valid covariance matrix
        determinant = esp_var * cec_var - cov**2

        invalid_cov = (determinant < 0) & ~mask
        n_invalid = invalid_cov.sum()

        print(f"Locations with invalid covariance matrix: {n_invalid}")

        if n_invalid > 0:
            print("\nThis happens when |ρ × σ_ESP × σ_CEC| > √(σ²_ESP × σ²_CEC)")
            print("Common causes:")
            print("  - Very small variances causing numerical issues")
            print("  - Correlation coefficient estimated from different data")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    issues = []

    if (n_esp_negative + n_esp_zero + n_cec_negative + n_cec_zero) > 0:
        issues.append("• Negative or zero uncertainties detected")
        print("\n1. Fix negative/zero uncertainties:")
        print("   Set a minimum threshold (e.g., 0.1% for ESP, 0.01 for CEC)")

    if (np.isinf(esp_mean).sum() + np.isinf(cec_mean).sum() +
        np.isinf(esp_std).sum() + np.isinf(cec_std).sum()) > 0:
        issues.append("• Infinite values detected")
        print("\n2. Remove infinite values:")
        print("   Mask or replace with NoData")

    if (rel_unc_esp > 1.0).sum() > 0 or (rel_unc_cec > 1.0).sum() > 0:
        issues.append("• Very large relative uncertainties (>100%)")
        print("\n3. Consider using analytical method instead of Monte Carlo:")
        print("   Monte Carlo can be unstable with very large uncertainties")

    if len(issues) == 0:
        print("\nNo major data quality issues detected.")
        print("Sampling failures may be due to numerical precision.")
        print("Consider using the analytical method for better stability.")

    print("\n" + "="*70)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python diagnose_data_issues.py <esp_mean.tif> <esp_std.tif> <cec_mean.tif> <cec_std.tif>")
        sys.exit(1)

    diagnose_geotiff_quality(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
