"""
GeoTIFF Processing for Gypsum Requirement Calculation

This module provides functions to calculate gypsum requirements directly from GeoTIFF files,
preserving spatial reference information in the output.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path
from gypsum_requirement import calculate_gypsum_requirement


def calculate_gypsum_from_geotiff(
    esp_initial_tif,
    esp_initial_uncertainty_tif,
    cec_tif,
    cec_uncertainty_tif,
    output_dir,
    output_prefix="gypsum",
    esp_final=5.0,
    efficiency_factor=1.2,
    soil_depth=0.15,
    bulk_density=1.2,
    nodata_value=-9999
):
    """
    Calculate gypsum requirement from GeoTIFF files and save results as GeoTIFFs.
    
    Parameters
    ----------
    esp_initial_tif : str or Path
        Path to GeoTIFF file containing ESP initial measurements (%)
        
    esp_initial_uncertainty_tif : str or Path
        Path to GeoTIFF file containing ESP initial uncertainties (%)
        
    cec_tif : str or Path
        Path to GeoTIFF file containing CEC measurements (cmol_c/kg or meq/100g)

    cec_uncertainty_tif : str or Path
        Path to GeoTIFF file containing CEC uncertainties (cmol_c/kg)
        
    output_dir : str or Path
        Directory where output GeoTIFF files will be saved
        
    output_prefix : str, optional
        Prefix for output filenames, default = "gypsum"
        
    esp_final : float, optional
        Target final ESP (%), default = 5.0
        
    efficiency_factor : float, optional
        Ca-Na exchange efficiency factor (unitless), default = 1.2
        
    soil_depth : float, optional
        Soil depth to be treated (m), default = 0.15 (15 cm)
        
    bulk_density : float, optional
        Soil bulk density (Mg/m³), default = 1.2
        
    nodata_value : float, optional
        Value to use for NoData in output rasters, default = -9999
        
    Returns
    -------
    output_files : dict
        Dictionary containing paths to the three output GeoTIFF files:
        - 'gypsum_requirement': GR values (Mg/ha)
        - 'gypsum_uncertainty': GR uncertainties (Mg/ha)
        - 'no_gypsum_flag': Boolean flag (1 = no gypsum needed, 0 = gypsum needed)
        
    Raises
    ------
    ValueError
        If input GeoTIFFs do not have matching dimensions, CRS, or transform
        
    Examples
    --------
    >>> output_files = calculate_gypsum_from_geotiff(
    ...     'esp_initial.tif',
    ...     'esp_initial_unc.tif',
    ...     'cec.tif',
    ...     'cec_unc.tif',
    ...     'output_folder'
    ... )
    >>> print(f"Gypsum requirement saved to: {output_files['gypsum_requirement']}")
    """
    
    # Convert paths to Path objects
    esp_initial_tif = Path(esp_initial_tif)
    esp_initial_uncertainty_tif = Path(esp_initial_uncertainty_tif)
    cec_tif = Path(cec_tif)
    cec_uncertainty_tif = Path(cec_uncertainty_tif)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input GeoTIFFs
    print(f"Reading input GeoTIFF files...")
    with rasterio.open(esp_initial_tif) as src:
        esp_initial = src.read(1)  # Read first band
        profile = src.profile.copy()  # Save georeferencing info
        transform = src.transform
        crs = src.crs
        shape = esp_initial.shape
        
    with rasterio.open(esp_initial_uncertainty_tif) as src:
        esp_initial_uncertainty = src.read(1)
        _validate_georeference(src, shape, transform, crs, "ESP initial uncertainty")
        
    with rasterio.open(cec_tif) as src:
        cec = src.read(1)
        _validate_georeference(src, shape, transform, crs, "CEC")
        
    with rasterio.open(cec_uncertainty_tif) as src:
        cec_uncertainty = src.read(1)
        _validate_georeference(src, shape, transform, crs, "CEC uncertainty")
    
    print(f"Input raster shape: {shape}")
    print(f"CRS: {crs}")
    
    # Handle NoData values - convert to NaN for calculation
    esp_initial_mask = esp_initial == profile.get('nodata', nodata_value)
    cec_mask = cec == profile.get('nodata', nodata_value)
    
    # Create combined mask (NoData in any input = NoData in output)
    combined_mask = esp_initial_mask | cec_mask
    
    # Replace NoData with NaN for calculation
    esp_initial = np.where(esp_initial_mask, np.nan, esp_initial)
    esp_initial_uncertainty = np.where(esp_initial_mask, np.nan, esp_initial_uncertainty)
    cec = np.where(cec_mask, np.nan, cec)
    cec_uncertainty = np.where(cec_mask, np.nan, cec_uncertainty)
    
    # Calculate gypsum requirement
    print(f"Calculating gypsum requirement...")
    gypsum_req, gypsum_unc, no_gypsum_flag = calculate_gypsum_requirement(
        esp_initial,
        esp_initial_uncertainty,
        cec,
        cec_uncertainty,
        esp_final=esp_final,
        efficiency_factor=efficiency_factor,
        soil_depth=soil_depth,
        bulk_density=bulk_density
    )
    
    # Apply mask to outputs
    gypsum_req = np.where(combined_mask, nodata_value, gypsum_req)
    gypsum_unc = np.where(combined_mask, nodata_value, gypsum_unc)
    no_gypsum_flag = np.where(combined_mask, 255, no_gypsum_flag.astype(np.uint8))  # 255 for NoData in byte
    
    # Update profile for output files
    output_profile = profile.copy()
    output_profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw',
        nodata=nodata_value
    )
    
    # Define output file paths
    output_files = {
        'gypsum_requirement': output_dir / f"{output_prefix}_requirement_Mg_ha.tif",
        'gypsum_uncertainty': output_dir / f"{output_prefix}_uncertainty_Mg_ha.tif",
        'no_gypsum_flag': output_dir / f"{output_prefix}_no_application_flag.tif"
    }
    
    # Write gypsum requirement
    print(f"Writing output: {output_files['gypsum_requirement']}")
    with rasterio.open(output_files['gypsum_requirement'], 'w', **output_profile) as dst:
        dst.write(gypsum_req.astype(rasterio.float32), 1)
        dst.set_band_description(1, "Gypsum Requirement (Mg/ha)")
    
    # Write gypsum uncertainty
    print(f"Writing output: {output_files['gypsum_uncertainty']}")
    with rasterio.open(output_files['gypsum_uncertainty'], 'w', **output_profile) as dst:
        dst.write(gypsum_unc.astype(rasterio.float32), 1)
        dst.set_band_description(1, "Gypsum Requirement Uncertainty (Mg/ha)")
    
    # Write flag as byte (0 = gypsum needed, 1 = no gypsum needed, 255 = NoData)
    flag_profile = output_profile.copy()
    flag_profile.update(dtype=rasterio.uint8, nodata=255)
    print(f"Writing output: {output_files['no_gypsum_flag']}")
    with rasterio.open(output_files['no_gypsum_flag'], 'w', **flag_profile) as dst:
        dst.write(no_gypsum_flag, 1)
        dst.set_band_description(1, "No Gypsum Needed Flag (1=no gypsum, 0=gypsum needed)")
    
    # Print summary statistics
    valid_mask = ~combined_mask
    if np.any(valid_mask):
        gypsum_values = gypsum_req[valid_mask & (gypsum_req != nodata_value)]
        if len(gypsum_values) > 0:
            print("\n" + "="*60)
            print("SUMMARY STATISTICS")
            print("="*60)
            print(f"Total pixels: {np.sum(valid_mask)}")
            print(f"Pixels requiring gypsum: {np.sum(valid_mask & ~no_gypsum_flag)}")
            print(f"Pixels below ESP threshold: {np.sum(valid_mask & no_gypsum_flag)}")
            print(f"Mean gypsum rate: {np.mean(gypsum_values):.2f} Mg/ha")
            print(f"Median gypsum rate: {np.median(gypsum_values):.2f} Mg/ha")
            print(f"Min gypsum rate: {np.min(gypsum_values):.2f} Mg/ha")
            print(f"Max gypsum rate: {np.max(gypsum_values):.2f} Mg/ha")
            print(f"Std dev: {np.std(gypsum_values):.2f} Mg/ha")
            print("="*60 + "\n")
    
    print("Processing complete!")
    return output_files


def _validate_georeference(src, expected_shape, expected_transform, expected_crs, name):
    """
    Validate that a raster has matching georeferencing with the reference.
    
    Parameters
    ----------
    src : rasterio.DatasetReader
        Open rasterio dataset to validate
    expected_shape : tuple
        Expected raster shape
    expected_transform : Affine
        Expected affine transform
    expected_crs : CRS
        Expected coordinate reference system
    name : str
        Name of the raster for error messages
    """
    if src.shape != expected_shape:
        raise ValueError(
            f"{name} has different dimensions {src.shape} "
            f"than ESP initial {expected_shape}"
        )
    
    if src.transform != expected_transform:
        raise ValueError(
            f"{name} has different geotransform than ESP initial"
        )
    
    if src.crs != expected_crs:
        raise ValueError(
            f"{name} has different CRS {src.crs} than ESP initial {expected_crs}"
        )


def batch_process_fields(field_config, output_base_dir, **kwargs):
    """
    Process multiple fields in batch.
    
    Parameters
    ----------
    field_config : dict
        Dictionary where keys are field names and values are dicts containing:
        - 'esp_initial': path to ESP initial GeoTIFF
        - 'esp_initial_uncertainty': path to ESP uncertainty GeoTIFF
        - 'cec': path to CEC GeoTIFF
        - 'cec_uncertainty': path to CEC uncertainty GeoTIFF
        
    output_base_dir : str or Path
        Base directory for outputs (subdirectories created per field)
        
    **kwargs : optional
        Additional parameters passed to calculate_gypsum_from_geotiff
        
    Returns
    -------
    results : dict
        Dictionary with field names as keys and output file dicts as values
        
    Examples
    --------
    >>> config = {
    ...     'field_1': {
    ...         'esp_initial': 'field1_esp.tif',
    ...         'esp_initial_uncertainty': 'field1_esp_unc.tif',
    ...         'cec': 'field1_cec.tif',
    ...         'cec_uncertainty': 'field1_cec_unc.tif'
    ...     },
    ...     'field_2': {...}
    ... }
    >>> results = batch_process_fields(config, 'outputs')
    """
    output_base_dir = Path(output_base_dir)
    results = {}
    
    for field_name, paths in field_config.items():
        print(f"\n{'='*60}")
        print(f"Processing field: {field_name}")
        print(f"{'='*60}\n")
        
        output_dir = output_base_dir / field_name
        
        try:
            output_files = calculate_gypsum_from_geotiff(
                esp_initial_tif=paths['esp_initial'],
                esp_initial_uncertainty_tif=paths['esp_initial_uncertainty'],
                cec_tif=paths['cec'],
                cec_uncertainty_tif=paths['cec_uncertainty'],
                output_dir=output_dir,
                output_prefix=field_name,
                **kwargs
            )
            results[field_name] = output_files
            
        except Exception as e:
            print(f"ERROR processing {field_name}: {str(e)}")
            results[field_name] = None
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python gypsum_geotiff.py <esp_initial.tif> <esp_unc.tif> <cec.tif> <cec_unc.tif> [output_dir]")
        sys.exit(1)
    
    output_dir = sys.argv[5] if len(sys.argv) > 5 else "gypsum_output"
    
    output_files = calculate_gypsum_from_geotiff(
        esp_initial_tif=sys.argv[1],
        esp_initial_uncertainty_tif=sys.argv[2],
        cec_tif=sys.argv[3],
        cec_uncertainty_tif=sys.argv[4],
        output_dir=output_dir
    )
    
    print("\nOutput files created:")
    for key, path in output_files.items():
        print(f"  {key}: {path}")
