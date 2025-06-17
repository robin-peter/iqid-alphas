#!/usr/bin/env python3
"""
Quantitative corrections and calibration parameters for UCSF Ac-225 iQID data processing.

This module contains the quantitative correction factors extracted from the 
UCSF quantitative notes document.

Author: Wookjin Choi <wookjin.choi@jefferson.edu>
Date: June 2025
"""

import numpy as np
from typing import Dict, Any, Tuple

class UCSFQuantitativeCorrections:
    """
    Quantitative corrections and calibration parameters for UCSF Ac-225 iQID data.
    
    Based on quantitative_notes.docx from UCSF Ac-225 murine sequential/3D dataset.
    """
    
    def __init__(self):
        """Initialize quantitative correction parameters."""
        
        # Core calibration parameters from quantitative notes
        self.geometric_efficiency = 0.5
        self.spatial_pileup_loss_25fps = 0.238  # 23.8% loss at 25 FPS
        self.frame_rate = 25.0  # FPS
        self.slice_thickness_cm = 10.0  # cm
        self.effective_pixel_size_um = 39.0  # micrometers
        self.ac225_to_alpha_ratio = 0.25  # Actual 225Ac is ~1/4 the indicated value
        
        # Derived correction factors
        self.efficiency_correction = 1.0 / self.geometric_efficiency
        self.pileup_correction = 1.0 / (1.0 - self.spatial_pileup_loss_25fps)
        self.combined_correction = self.efficiency_correction * self.pileup_correction
        
    def get_calibration_parameters(self) -> Dict[str, Any]:
        """Get all calibration parameters."""
        return {
            'geometric_efficiency': self.geometric_efficiency,
            'spatial_pileup_loss_25fps': self.spatial_pileup_loss_25fps,
            'frame_rate_fps': self.frame_rate,
            'slice_thickness_cm': self.slice_thickness_cm,
            'effective_pixel_size_um': self.effective_pixel_size_um,
            'ac225_to_alpha_ratio': self.ac225_to_alpha_ratio,
            'efficiency_correction_factor': self.efficiency_correction,
            'pileup_correction_factor': self.pileup_correction,
            'combined_correction_factor': self.combined_correction
        }
    
    def apply_quantitative_corrections(self, measured_activity: np.ndarray, 
                                     frame_rate: float = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply quantitative corrections to measured iQID activity.
        
        Formula from quantitative notes:
        True activity ∝ (A)(1/0.5)/(1 – 0.238)
        
        Args:
            measured_activity: Raw measured activity counts
            frame_rate: Frame rate in FPS (uses default 25 FPS if None)
            
        Returns:
            corrected_activity: Activity corrected for efficiency and pileup
            correction_metadata: Dictionary of applied corrections
        """
        if frame_rate is None:
            frame_rate = self.frame_rate
        
        # Apply efficiency and pileup corrections
        corrected_activity = measured_activity * self.combined_correction
        
        # Frame rate correction (if different from calibration frame rate)
        frame_rate_correction = 1.0
        if frame_rate != self.frame_rate:
            # Frame rate correction should be applied (see paper reference)
            # For now, using linear scaling - this may need refinement based on the paper
            frame_rate_correction = frame_rate / self.frame_rate
            corrected_activity *= frame_rate_correction
        
        correction_metadata = {
            'geometric_efficiency_correction': self.efficiency_correction,
            'spatial_pileup_correction': self.pileup_correction,
            'frame_rate_correction': frame_rate_correction,
            'combined_correction_factor': self.combined_correction * frame_rate_correction,
            'input_frame_rate': frame_rate,
            'calibration_frame_rate': self.frame_rate
        }
        
        return corrected_activity, correction_metadata
    
    def convert_to_ac225_activity(self, alpha_activity: np.ndarray) -> np.ndarray:
        """
        Convert alpha particle activity to actual Ac-225 activity.
        
        From notes: "actual 225Ac is approximately ¼ the indicated value"
        
        Args:
            alpha_activity: Alpha particle activity
            
        Returns:
            ac225_activity: Actual Ac-225 activity
        """
        return alpha_activity * self.ac225_to_alpha_ratio
    
    def get_spatial_calibration(self) -> Dict[str, float]:
        """Get spatial calibration parameters."""
        return {
            'pixel_size_um': self.effective_pixel_size_um,
            'pixel_size_mm': self.effective_pixel_size_um / 1000.0,
            'slice_thickness_cm': self.slice_thickness_cm,
            'slice_thickness_mm': self.slice_thickness_cm * 10.0,
            'voxel_volume_mm3': (self.effective_pixel_size_um / 1000.0)**2 * (self.slice_thickness_cm * 10.0)
        }
    
    def calculate_activity_concentration(self, corrected_activity: np.ndarray, 
                                       units: str = 'Bq_per_mm3') -> np.ndarray:
        """
        Calculate activity concentration from corrected activity.
        
        Args:
            corrected_activity: Activity corrected for efficiency and pileup
            units: Output units ('Bq_per_mm3', 'kBq_per_cm3', 'mBq_per_mm3')
            
        Returns:
            activity_concentration: Activity per unit volume
        """
        spatial_cal = self.get_spatial_calibration()
        voxel_volume_mm3 = spatial_cal['voxel_volume_mm3']
        
        # Base calculation: activity per mm³
        concentration_bq_per_mm3 = corrected_activity / voxel_volume_mm3
        
        if units == 'Bq_per_mm3':
            return concentration_bq_per_mm3
        elif units == 'kBq_per_cm3':
            return concentration_bq_per_mm3 * 1000.0 / 1000.0  # Bq/mm³ to kBq/cm³
        elif units == 'mBq_per_mm3':
            return concentration_bq_per_mm3 * 1000.0
        else:
            raise ValueError(f"Unsupported units: {units}")
    
    def generate_correction_report(self, measured_activity: np.ndarray, 
                                 corrected_activity: np.ndarray,
                                 correction_metadata: Dict[str, float]) -> Dict[str, Any]:
        """Generate a comprehensive correction report."""
        
        spatial_cal = self.get_spatial_calibration()
        
        report = {
            'quantitative_corrections': {
                'source_document': 'quantitative_notes.docx - UCSF Ac-225 murine sequential/3D dataset',
                'calibration_parameters': self.get_calibration_parameters(),
                'applied_corrections': correction_metadata,
                'spatial_calibration': spatial_cal
            },
            'processing_summary': {
                'input_activity_range': {
                    'min': float(np.min(measured_activity)),
                    'max': float(np.max(measured_activity)),
                    'mean': float(np.mean(measured_activity)),
                    'total': float(np.sum(measured_activity))
                },
                'corrected_activity_range': {
                    'min': float(np.min(corrected_activity)),
                    'max': float(np.max(corrected_activity)),
                    'mean': float(np.mean(corrected_activity)),
                    'total': float(np.sum(corrected_activity))
                },
                'correction_factor_applied': correction_metadata['combined_correction_factor']
            },
            'notes': {
                'activity_type': 'alpha-particle activity (detected alpha particles)',
                'ac225_conversion': f'Actual 225Ac activity ≈ {self.ac225_to_alpha_ratio} × indicated value',
                'frame_rate_dependency': 'Corrections calibrated for 25 FPS acquisition',
                'spatial_resolution': f'{self.effective_pixel_size_um} µm effective pixel size',
                'slice_thickness': f'{self.slice_thickness_cm} cm slice thickness'
            }
        }
        
        return report

def load_quantitative_corrections() -> UCSFQuantitativeCorrections:
    """
    Load and return the UCSF quantitative corrections object.
    
    Returns:
        UCSFQuantitativeCorrections: Initialized corrections object
    """
    return UCSFQuantitativeCorrections()

# Example usage and testing
if __name__ == "__main__":
    # Test the quantitative corrections
    corrections = UCSFQuantitativeCorrections()
    
    print("🔬 UCSF Ac-225 Quantitative Corrections")
    print("=" * 50)
    
    # Display calibration parameters
    params = corrections.get_calibration_parameters()
    print("\n📊 Calibration Parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # Test with sample data
    print("\n🧪 Testing with sample data:")
    sample_activity = np.array([[100, 200, 150], [300, 250, 180], [120, 350, 220]])
    print(f"Sample measured activity:\n{sample_activity}")
    
    corrected_activity, metadata = corrections.apply_quantitative_corrections(sample_activity)
    print(f"\nCorrected activity:\n{corrected_activity}")
    print(f"\nCorrection factor applied: {metadata['combined_correction_factor']:.3f}")
    
    # Test Ac-225 conversion
    ac225_activity = corrections.convert_to_ac225_activity(corrected_activity)
    print(f"\nActual Ac-225 activity:\n{ac225_activity}")
    
    # Test spatial calibration
    spatial_cal = corrections.get_spatial_calibration()
    print(f"\n📏 Spatial Calibration:")
    for key, value in spatial_cal.items():
        print(f"   {key}: {value}")
    
    # Test activity concentration
    concentration = corrections.calculate_activity_concentration(corrected_activity, 'mBq_per_mm3')
    print(f"\nActivity concentration (mBq/mm³):\n{concentration}")
