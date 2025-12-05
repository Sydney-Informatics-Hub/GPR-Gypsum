"""
Gypsum Requirement Calculation with Uncertainty Propagation
Based on equation 8.2 from Agricultural Management of Sodic Soils (Oster et al.)

GR = 0.086 * F * Ds * ρb * CEC * (ESPi - ESPf)
Note: 0.086 is for CEC in cmol_c/kg (0.0086 would be for mmol_c/kg)
"""

import numpy as np


def calculate_gypsum_requirement(
    esp_initial,
    esp_initial_uncertainty,
    cec,
    cec_uncertainty,
    esp_final=5.0,
    efficiency_factor=1.2,
    soil_depth=0.15,
    bulk_density=1.2
):
    """
    Calculate gypsum requirement (GR) with uncertainty propagation.
    
    Parameters
    ----------
    esp_initial : numpy.ndarray
        Array of initial exchangeable sodium percentage (ESP) measurements (%)
        Shape: any dimensions (1D, 2D, 3D depending on spatial support)
        
    esp_initial_uncertainty : numpy.ndarray
        Array of uncertainties for ESP initial values (%)
        Shape: same as esp_initial
        
    cec : numpy.ndarray
        Array of cation exchange capacity measurements (cmol_c/kg)
        Shape: same as esp_initial

    cec_uncertainty : numpy.ndarray
        Array of uncertainties for CEC values (cmol_c/kg)
        Shape: same as esp_initial
        
    esp_final : float, optional
        Target final ESP (%), default = 5.0
        Set below sodicity threshold of 6%
        
    efficiency_factor : float, optional
        Ca-Na exchange efficiency factor (unitless), default = 1.2
        Typical range: 1.1 to 1.3
        
    soil_depth : float, optional
        Soil depth to be treated (m), default = 0.15 (15 cm)
        
    bulk_density : float, optional
        Soil bulk density (Mg/m³), default = 1.2
        
    Returns
    -------
    gypsum_requirement : numpy.ndarray
        Array of gypsum requirement values (Mg/ha)
        Shape: same as input arrays
        Values set to 0 where ESP_initial <= ESP_final (no gypsum needed)
        
    gypsum_requirement_uncertainty : numpy.ndarray
        Array of propagated uncertainties for GR (Mg/ha)
        Shape: same as input arrays
        
    flags : numpy.ndarray (boolean)
        Boolean array indicating locations where ESP_initial <= ESP_final
        True = no gypsum required, False = gypsum required
        Shape: same as input arrays
        
    Notes
    -----
    - Uncertainty propagation assumes ESP and CEC are independent variables
    - Uses standard error propagation formula
    - Conversion factor 0.086 converts units to Mg/ha (for CEC in cmol_c/kg)
    
    Examples
    --------
    >>> esp = np.array([15.0, 25.0, 4.0, 30.0])
    >>> esp_unc = np.array([1.5, 2.0, 0.5, 2.5])
    >>> cec = np.array([200, 250, 180, 300])
    >>> cec_unc = np.array([20, 25, 18, 30])
    >>> gr, gr_unc, flags = calculate_gypsum_requirement(esp, esp_unc, cec, cec_unc)
    """
    
    # Validate input shapes
    if not (esp_initial.shape == esp_initial_uncertainty.shape == 
            cec.shape == cec_uncertainty.shape):
        raise ValueError("All input arrays must have the same shape")
    
    # Conversion factor from equation (for CEC in cmol_c/kg)
    conversion_factor = 0.086
    
    # Calculate ESP difference
    esp_diff = esp_initial - esp_final
    
    # Flag locations where gypsum is not required (ESP_initial <= ESP_final)
    no_gypsum_needed = esp_initial <= esp_final
    
    # Calculate gypsum requirement
    # GR = 0.086 * F * Ds * ρb * CEC * (ESPi - ESPf)
    gypsum_requirement = (conversion_factor * 
                          efficiency_factor * 
                          soil_depth * 
                          bulk_density * 
                          cec * 
                          esp_diff)
    
    # Set GR to 0 where not needed
    gypsum_requirement = np.where(no_gypsum_needed, 0.0, gypsum_requirement)
    
    # Uncertainty propagation
    # For GR = k * CEC * (ESPi - ESPf), where k = 0.086 * F * Ds * ρb
    # δGR = sqrt[(∂GR/∂CEC * δCEC)² + (∂GR/∂ESPi * δESPi)²]
    # ∂GR/∂CEC = k * (ESPi - ESPf)
    # ∂GR/∂ESPi = k * CEC

    k = conversion_factor * efficiency_factor * soil_depth * bulk_density
    
    # Partial derivatives
    partial_cec = k * esp_diff
    partial_esp = k * cec
    
    # Propagated uncertainty
    gypsum_uncertainty = np.sqrt(
        (partial_cec * cec_uncertainty)**2 + 
        (partial_esp * esp_initial_uncertainty)**2
    )
    
    # Set uncertainty to 0 where gypsum is not needed
    gypsum_uncertainty = np.where(no_gypsum_needed, 0.0, gypsum_uncertainty)
    
    return gypsum_requirement, gypsum_uncertainty, no_gypsum_needed


def summarize_gypsum_application(gypsum_requirement, flags, area_per_cell=None):
    """
    Generate summary statistics for gypsum application.
    
    Parameters
    ----------
    gypsum_requirement : numpy.ndarray
        Array of gypsum requirement values (Mg/ha)
        
    flags : numpy.ndarray (boolean)
        Boolean array indicating locations where no gypsum is required
        
    area_per_cell : float, optional
        Area represented by each cell (ha). If provided, calculates total gypsum needed.
        
    Returns
    -------
    summary : dict
        Dictionary containing summary statistics
    """
    
    # Areas requiring gypsum
    requires_gypsum = ~flags
    
    summary = {
        'total_cells': flags.size,
        'cells_requiring_gypsum': np.sum(requires_gypsum),
        'cells_no_gypsum_needed': np.sum(flags),
        'percent_requiring_gypsum': 100 * np.sum(requires_gypsum) / flags.size,
        'mean_gypsum_rate': np.mean(gypsum_requirement[requires_gypsum]) if np.any(requires_gypsum) else 0,
        'median_gypsum_rate': np.median(gypsum_requirement[requires_gypsum]) if np.any(requires_gypsum) else 0,
        'min_gypsum_rate': np.min(gypsum_requirement[requires_gypsum]) if np.any(requires_gypsum) else 0,
        'max_gypsum_rate': np.max(gypsum_requirement[requires_gypsum]) if np.any(requires_gypsum) else 0,
        'std_gypsum_rate': np.std(gypsum_requirement[requires_gypsum]) if np.any(requires_gypsum) else 0,
    }
    
    if area_per_cell is not None:
        summary['total_gypsum_needed_Mg'] = np.sum(gypsum_requirement) * area_per_cell
        summary['total_area_ha'] = flags.size * area_per_cell
        summary['area_requiring_gypsum_ha'] = np.sum(requires_gypsum) * area_per_cell
    
    return summary
