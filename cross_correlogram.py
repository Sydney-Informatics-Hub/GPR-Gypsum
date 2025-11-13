"""
Cross-Correlogram Calculation for ESP and CEC from GeoTIFF files

This script computes the cross-correlation between ESP and CEC as a function
of spatial distance, which is useful for understanding spatial relationships
and for geostatistical modeling.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import json


def compute_cross_correlogram(X, var1, var2, n_bins=15, max_dist=None, max_points=None):
    """
    Compute cross-correlation as a function of distance

    Parameters
    ----------
    X : numpy.ndarray
        Coordinate array of shape (n, 2) containing (x, y) locations
    var1 : numpy.ndarray
        First variable values (centered), shape (n,)
    var2 : numpy.ndarray
        Second variable values (centered), shape (n,)
    n_bins : int, optional
        Number of distance bins, default = 15
    max_dist : float, optional
        Maximum distance to consider. If None, uses 75th percentile of distances
    max_points : int, optional
        Maximum number of points to use. If n > max_points, randomly subsample.
        Default = None (use all points)

    Returns
    -------
    bin_centers : numpy.ndarray
        Center distances for each bin
    cross_corr : numpy.ndarray
        Cross-correlation values for each bin
    counts : numpy.ndarray
        Number of pairs in each bin
    """
    # Subsample if needed to avoid memory issues
    n = len(X)
    if max_points is not None and n > max_points:
        print(f"  Subsampling from {n:,} to {max_points:,} points to avoid memory issues")
        np.random.seed(42)  # For reproducibility
        idx = np.random.choice(n, size=max_points, replace=False)
        X = X[idx]
        var1 = var1[idx]
        var2 = var2[idx]

    # Compute all pairwise distances
    distances = squareform(pdist(X))
    
    # Compute all pairwise cross-products
    n = len(var1)
    cross_products = np.outer(var1, var2)  # (n, n)
    
    # Flatten upper triangle (avoid double counting)
    triu_idx = np.triu_indices(n, k=1)
    dists_flat = distances[triu_idx]
    cross_flat = cross_products[triu_idx]
    
    # Set max distance if not provided
    if max_dist is None:
        max_dist = np.percentile(dists_flat, 75)  # Use 75th percentile
    
    # Create distance bins
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute mean cross-product in each bin
    cross_corr = []
    counts = []
    
    for i in range(n_bins):
        mask = (dists_flat >= bin_edges[i]) & (dists_flat < bin_edges[i+1])
        if mask.sum() > 0:
            cross_corr.append(cross_flat[mask].mean())
            counts.append(mask.sum())
        else:
            cross_corr.append(np.nan)
            counts.append(0)
    
    # Normalize by variances to get correlation
    var1_std = var1.std()
    var2_std = var2.std()
    cross_corr = np.array(cross_corr) / (var1_std * var2_std)
    
    return bin_centers, cross_corr, np.array(counts)


def compute_cross_correlogram_from_geotiff(
    esp_tif,
    cec_tif,
    output_dir=None,
    output_prefix="cross_correlogram",
    n_bins=20,
    max_dist=None,
    max_points=10000,
    plot=True,
    save_data=True
):
    """
    Compute cross-correlogram between ESP and CEC from GeoTIFF files.

    Parameters
    ----------
    esp_tif : str or Path
        Path to GeoTIFF file containing ESP values (%)

    cec_tif : str or Path
        Path to GeoTIFF file containing CEC values (mmol_c/kg)

    output_dir : str or Path, optional
        Directory where output files will be saved. If None, uses current directory

    output_prefix : str, optional
        Prefix for output filenames, default = "cross_correlogram"

    n_bins : int, optional
        Number of distance bins, default = 20

    max_dist : float, optional
        Maximum distance to consider (in map units). If None, uses 75th percentile

    max_points : int, optional
        Maximum number of points to use for correlation computation.
        If total valid pixels > max_points, randomly subsample to avoid memory issues.
        Default = 10000 (sufficient for most correlation estimates)

    plot : bool, optional
        Whether to create and save a plot, default = True

    save_data : bool, optional
        Whether to save correlogram data to CSV, default = True
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'lags': bin center distances
        - 'cross_correlation': cross-correlation values
        - 'counts': number of pairs per bin
        - 'esp_mean': mean ESP value
        - 'cec_mean': mean CEC value
        - 'esp_std': ESP standard deviation
        - 'cec_std': CEC standard deviation
        - 'n_points': number of valid data points used
        
    Examples
    --------
    >>> results = compute_cross_correlogram_from_geotiff(
    ...     'esp_field.tif',
    ...     'cec_field.tif',
    ...     output_dir='results',
    ...     n_bins=20
    ... )
    """
    # Convert paths
    esp_tif = Path(esp_tif)
    cec_tif = Path(cec_tif)
    
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading GeoTIFF files...")
    print(f"  ESP: {esp_tif}")
    print(f"  CEC: {cec_tif}")
    
    # Read ESP data
    with rasterio.open(esp_tif) as src:
        esp_data = src.read(1)
        transform = src.transform
        nodata_esp = src.nodata
        shape = esp_data.shape
        
        # Get coordinates for each pixel
        rows, cols = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
        coords = np.column_stack([xs, ys])
        
    # Read CEC data
    with rasterio.open(cec_tif) as src:
        cec_data = src.read(1)
        nodata_cec = src.nodata
        
        if cec_data.shape != shape:
            raise ValueError(
                f"ESP and CEC rasters have different shapes: "
                f"{shape} vs {cec_data.shape}"
            )
    
    # Flatten arrays
    esp_flat = esp_data.flatten()
    cec_flat = cec_data.flatten()
    
    # Create mask for valid data (no NoData values)
    valid_mask = np.ones(len(esp_flat), dtype=bool)
    
    if nodata_esp is not None:
        valid_mask &= (esp_flat != nodata_esp)
    if nodata_cec is not None:
        valid_mask &= (cec_flat != nodata_cec)
    
    # Also remove NaN and Inf values
    valid_mask &= np.isfinite(esp_flat) & np.isfinite(cec_flat)
    
    # Filter to valid data
    X = coords[valid_mask]
    ESP = esp_flat[valid_mask]
    CEC = cec_flat[valid_mask]
    
    n_valid = len(ESP)
    n_total = len(esp_flat)
    
    print(f"\nData summary:")
    print(f"  Total pixels: {n_total}")
    print(f"  Valid pixels: {n_valid} ({100*n_valid/n_total:.1f}%)")
    print(f"  ESP range: [{ESP.min():.2f}, {ESP.max():.2f}] %")
    print(f"  CEC range: [{CEC.min():.2f}, {CEC.max():.2f}] mmol_c/kg")
    
    # Center the data (subtract mean)
    ESP_mean = ESP.mean()
    CEC_mean = CEC.mean()
    ESP_std = ESP.std()
    CEC_std = CEC.std()
    
    ESP_centered = ESP - ESP_mean
    CEC_centered = CEC - CEC_mean
    
    print(f"  ESP mean: {ESP_mean:.2f} %, std: {ESP_std:.2f} %")
    print(f"  CEC mean: {CEC_mean:.2f} mmol_c/kg, std: {CEC_std:.2f} mmol_c/kg")
    
    # Compute cross-correlogram
    print(f"\nComputing cross-correlogram with {n_bins} bins...")
    lags, cross_corr, counts = compute_cross_correlogram(
        X, ESP_centered, CEC_centered, n_bins=n_bins, max_dist=max_dist, max_points=max_points
    )
    
    print(f"  Distance range: [0, {lags.max():.1f}] map units")
    print(f"  Cross-correlation range: [{np.nanmin(cross_corr):.3f}, {np.nanmax(cross_corr):.3f}]")
    
    # Prepare results dictionary
    results = {
        'lags': lags,
        'cross_correlation': cross_corr,
        'counts': counts,
        'esp_mean': ESP_mean,
        'cec_mean': CEC_mean,
        'esp_std': ESP_std,
        'cec_std': CEC_std,
        'n_points': n_valid,
        'n_bins': n_bins,
        'max_dist': lags.max()
    }
    
    # Save data to CSV
    if save_data:
        csv_path = output_dir / f"{output_prefix}_data.csv"
        print(f"\nSaving correlogram data to: {csv_path}")
        
        import pandas as pd
        df = pd.DataFrame({
            'lag_distance': lags,
            'cross_correlation': cross_corr,
            'n_pairs': counts
        })
        df.to_csv(csv_path, index=False)
        
        # Save metadata
        metadata_path = output_dir / f"{output_prefix}_metadata.json"
        metadata = {
            'esp_mean': float(ESP_mean),
            'esp_std': float(ESP_std),
            'cec_mean': float(CEC_mean),
            'cec_std': float(CEC_std),
            'n_points': int(n_valid),
            'n_bins': int(n_bins),
            'max_distance': float(lags.max()),
            'esp_file': str(esp_tif),
            'cec_file': str(cec_tif)
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saving metadata to: {metadata_path}")
    
    # Create plot
    if plot:
        plot_path = output_dir / f"{output_prefix}_plot.png"
        print(f"\nCreating plot: {plot_path}")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Cross-correlation vs distance
        ax1.plot(lags, cross_corr, 'o-', linewidth=2, markersize=6, color='#2E86AB')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Distance (map units)', fontsize=12)
        ax1.set_ylabel('Cross-correlation', fontsize=12)
        ax1.set_title('ESP-CEC Cross-Correlogram', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([np.nanmin(cross_corr) - 0.1, np.nanmax(cross_corr) + 0.1])
        
        # Plot 2: Number of pairs per bin
        ax2.bar(lags, counts, width=lags[1]-lags[0], color='#A23B72', alpha=0.7)
        ax2.set_xlabel('Distance (map units)', fontsize=12)
        ax2.set_ylabel('Number of pairs', fontsize=12)
        ax2.set_title('Sample Size per Distance Bin', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved successfully")
    
    print("\nProcessing complete!")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python cross_correlogram.py <esp.tif> <cec.tif> [output_dir] [n_bins] [max_points]")
        print("\nExample:")
        print("  python cross_correlogram.py esp_field.tif cec_field.tif results 20 10000")
        print("\nNote: max_points (default=10000) limits memory usage for large rasters")
        sys.exit(1)

    esp_file = sys.argv[1]
    cec_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "correlogram_output"
    n_bins = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    max_points = int(sys.argv[5]) if len(sys.argv) > 5 else 10000

    results = compute_cross_correlogram_from_geotiff(
        esp_tif=esp_file,
        cec_tif=cec_file,
        output_dir=output_dir,
        n_bins=n_bins,
        max_points=max_points,
        plot=True,
        save_data=True
    )
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Number of valid points: {results['n_points']}")
    print(f"ESP: mean = {results['esp_mean']:.2f}, std = {results['esp_std']:.2f}")
    print(f"CEC: mean = {results['cec_mean']:.2f}, std = {results['cec_std']:.2f}")
    print(f"Max correlation: {np.nanmax(results['cross_correlation']):.3f}")
    print(f"Min correlation: {np.nanmin(results['cross_correlation']):.3f}")
    print("="*60)
