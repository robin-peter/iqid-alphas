#!/usr/bin/env python3
"""
UCSF Data Loader and Sample Matcher

This module provides utilities to navigate the UCSF dataset structure,
match samples between H&E and iQID data, and load the appropriate data files.

Structure:
- H&E: DataPush1/HE/{3D|Sequential sections (10um)}/{kidney|tumor}/{SampleID}/P{N}{L|R}.tif
- iQID (processed): DataPush1/iQID/{3D|Sequential sections}/{kidney|tumor}/{SampleID}/mBq_corr_{N}.tif  
- iQID (raw): ReUpload/iQID_reupload/iQID/{3D|Sequential scans}/{kidney|tumor}/{SampleID}/0_{SampleID}_iqid_event_image.tif

Author: Wookjin Choi <wookjin.choi@jefferson.edu>
Date: June 2025
"""

import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class UCSFDataMatcher:
    """
    Utility class for matching and loading UCSF H&E and iQID samples.
    """
    
    def __init__(self, base_path: str):
        """
        Initialize the data matcher.
        
        Args:
            base_path: Base path to UCSF data directory
        """
        self.base_path = Path(base_path)
        self.sample_matches = {}
        self.he_samples = {}
        self.iqid_samples = {}
        self._scan_available_samples()
    
    def _scan_available_samples(self):
        """Scan the data directory to find available samples."""
        logger.info("Scanning UCSF data directory for available samples...")
        
        # Find H&E samples
        he_samples = self._find_he_samples()
        logger.info(f"Found {len(he_samples)} H&E samples")
        
        # Find iQID samples  
        iqid_samples = self._find_iqid_samples()
        logger.info(f"Found {len(iqid_samples)} iQID samples")
        
        # Match samples between H&E and iQID
        self.sample_matches = self._match_samples(he_samples, iqid_samples)
        logger.info(f"Successfully matched {len(self.sample_matches)} sample pairs")
    
    def _find_he_samples(self) -> Dict[str, Dict]:
        """Find all available H&E samples."""
        he_samples = {}
        
        # Check DataPush1 H&E data
        he_base = self.base_path / "DataPush1" / "HE"
        
        for scan_type in ["3D", "Sequential sections (10um)"]:
            scan_path = he_base / scan_type
            if not scan_path.exists():
                continue
                
            for tissue_type in ["kidney", "tumor"]:
                tissue_path = scan_path / tissue_type
                if not tissue_path.exists():
                    continue
                
                # Find sample directories
                for sample_dir in tissue_path.iterdir():
                    if not sample_dir.is_dir():
                        continue
                    
                    sample_id = sample_dir.name
                    # Parse sample ID (e.g., D1M1_L, D7M2_R)
                    match = re.match(r'(D\d+M\d+)_([LR])', sample_id)
                    if not match:
                        continue
                    
                    base_id, side = match.groups()
                    
                    # Find image files
                    image_files = list(sample_dir.glob("P*.tif"))
                    
                    key = f"{base_id}_{side}"
                    he_samples[key] = {
                        'sample_id': sample_id,
                        'base_id': base_id,
                        'side': side,
                        'scan_type': scan_type,
                        'tissue_type': tissue_type,
                        'path': sample_dir,
                        'files': sorted([f.name for f in image_files]),
                        'file_count': len(image_files)
                    }
        
        return he_samples
    
    def _find_iqid_samples(self) -> Dict[str, Dict]:
        """Find all available iQID samples."""
        iqid_samples = {}
        
        # Check both DataPush1 (processed) and ReUpload (raw+processed)
        locations = [
            ("DataPush1", "DataPush1/iQID", ["3D", "Sequential sections"]),
            ("ReUpload", "ReUpload/iQID_reupload/iQID", ["3D", "Sequential scans"])
        ]
        
        for location_name, location_path, scan_types in locations:
            iqid_base = self.base_path / location_path
            if not iqid_base.exists():
                continue
            
            for scan_type in scan_types:
                scan_path = iqid_base / scan_type
                if not scan_path.exists():
                    continue
                    
                for tissue_type in ["kidney", "tumor"]:
                    tissue_path = scan_path / tissue_type
                    if not tissue_path.exists():
                        continue
                    
                    # Find sample directories
                    for sample_dir in tissue_path.iterdir():
                        if not sample_dir.is_dir():
                            continue
                        
                        sample_id = sample_dir.name
                        # Parse iQID sample ID (e.g., D1M1(P1)_L, D7M2(P2)_R)
                        match = re.match(r'(D\d+M\d+)(?:\(P\d+\))?_([LR])', sample_id)
                        if not match:
                            continue
                        
                        base_id, side = match.groups()
                        
                        # Determine data types available
                        data_types = {}
                        
                        if location_name == "ReUpload":
                            # Raw data
                            raw_files = list(sample_dir.glob("0_*_iqid_event_image.tif"))
                            if raw_files:
                                data_types['raw'] = raw_files[0]
                            
                            # Segmented data
                            seg_dir = sample_dir / "1_segmented"
                            if seg_dir.exists():
                                seg_files = list(seg_dir.glob("mBq_*.tif"))
                                data_types['segmented'] = sorted(seg_files)
                            
                            # Aligned data
                            aligned_dir = sample_dir / "2_aligned"
                            if aligned_dir.exists():
                                aligned_files = list(aligned_dir.glob("mBq_corr_*.tif"))
                                data_types['aligned'] = sorted(aligned_files)
                        
                        else:  # DataPush1
                            # Processed/aligned data
                            aligned_files = list(sample_dir.glob("mBq_corr_*.tif"))
                            if aligned_files:
                                data_types['aligned'] = sorted(aligned_files)
                        
                        key = f"{base_id}_{side}_{location_name.lower()}"
                        iqid_samples[key] = {
                            'sample_id': sample_id,
                            'base_id': base_id,
                            'side': side,
                            'scan_type': scan_type,
                            'tissue_type': tissue_type,
                            'location': location_name,
                            'path': sample_dir,
                            'data_types': data_types
                        }
        
        return iqid_samples
    
    def _match_samples(self, he_samples: Dict, iqid_samples: Dict) -> Dict[str, Dict]:
        """Match H&E and iQID samples based on sample IDs."""
        matches = {}
        
        for he_key, he_info in he_samples.items():
            base_id = he_info['base_id']
            side = he_info['side']
            
            # Find matching iQID samples
            iqid_matches = {}
            
            # Look for exact matches in both locations
            for iqid_key, iqid_info in iqid_samples.items():
                if (iqid_info['base_id'] == base_id and 
                    iqid_info['side'] == side):
                    
                    location = iqid_info['location']
                    iqid_matches[location.lower()] = iqid_info
            
            if iqid_matches:
                matches[he_key] = {
                    'he': he_info,
                    'iqid': iqid_matches
                }
        
        return matches
    
    def get_available_samples(self) -> List[str]:
        """Get list of available matched samples."""
        return list(self.sample_matches.keys())
    
    def get_all_he_samples(self) -> List[str]:
        """Get list of all available H&E samples (regardless of iQID matching)."""
        return list(self.he_samples.keys())
    
    def get_all_iqid_samples(self) -> List[str]:
        """Get list of all available iQID samples (regardless of H&E matching)."""
        return list(self.iqid_samples.keys())
    
    def get_sample_counts(self) -> Dict[str, int]:
        """Get counts of different sample types."""
        return {
            'total_he_samples': len(self.he_samples),
            'total_iqid_samples': len(self.iqid_samples),
            'matched_pairs': len(self.sample_matches),
            'unmatched_he_samples': len(self.he_samples) - len(self.sample_matches),
            'unmatched_iqid_samples': len(self.iqid_samples) - len([k for k in self.iqid_samples.keys() if any(k.startswith(m.split('_')[0]) for m in self.sample_matches.keys())])
        }
    
    def get_sample_info(self, sample_key: str) -> Optional[Dict]:
        """Get detailed information about a specific sample."""
        return self.sample_matches.get(sample_key)
    
    def get_he_sample_info(self, sample_key: str) -> Optional[Dict]:
        """Get H&E sample information."""
        return self.he_samples.get(sample_key)
    
    def get_iqid_sample_info(self, sample_key: str) -> Optional[Dict]:
        """Get iQID sample information."""
        return self.iqid_samples.get(sample_key)
    
    def load_he_data(self, sample_key: str) -> Optional[Dict[str, Any]]:
        """
        Load H&E data for a specific sample.
        
        Args:
            sample_key: Sample key (e.g., 'D1M1_L')
            
        Returns:
            Dictionary containing H&E data and metadata
        """
        if sample_key not in self.sample_matches:
            logger.warning(f"Sample {sample_key} not found")
            return None
        
        he_info = self.sample_matches[sample_key]['he']
        he_path = he_info['path']
        
        # Load all H&E images for this sample
        images = []
        metadata = {
            'sample_id': he_info['sample_id'],
            'tissue_type': he_info['tissue_type'],
            'scan_type': he_info['scan_type'],
            'file_count': he_info['file_count'],
            'files': he_info['files']
        }
        
        try:
            # For now, return file paths (actual loading would use PIL/cv2/tifffile)
            for filename in he_info['files']:
                file_path = he_path / filename
                if file_path.exists():
                    images.append(str(file_path))
            
            return {
                'images': images,
                'metadata': metadata,
                'path': str(he_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to load H&E data for {sample_key}: {e}")
            return None
    
    def load_he_data_by_key(self, sample_key: str) -> Optional[Dict[str, Any]]:
        """
        Load H&E data by sample key (works for any H&E sample, not just matched pairs).
        
        Args:
            sample_key: H&E sample key (e.g., 'D7M1_L')
            
        Returns:
            Dictionary containing H&E data and metadata
        """
        if sample_key not in self.he_samples:
            logger.warning(f"H&E sample {sample_key} not found")
            return None
        
        he_info = self.he_samples[sample_key]
        he_path = he_info['path']
        
        # Load all H&E images for this sample
        metadata = {
            'sample_id': he_info['sample_id'],
            'tissue_type': he_info['tissue_type'],
            'scan_type': he_info['scan_type'],
            'file_count': he_info['file_count'],
            'files': he_info['files'],
            'data_type': 'he_only'  # Mark as H&E-only sample
        }
        
        try:
            # For now, return file paths (actual loading would use PIL/cv2/tifffile)
            file_paths = []
            for filename in he_info['files']:
                file_path = he_path / filename
                if file_path.exists():
                    file_paths.append(str(file_path))
            
            return {
                'images': file_paths,
                'metadata': metadata,
                'source_path': str(he_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to load H&E data for {sample_key}: {e}")
            return None
    
    def load_iqid_data(self, sample_key: str, data_type: str = 'aligned', 
                       location: str = 'reupload') -> Optional[Dict[str, Any]]:
        """
        Load iQID data for a specific sample.
        
        Args:
            sample_key: Sample key (e.g., 'D1M1_L')
            data_type: Type of data ('raw', 'segmented', 'aligned')
            location: Data location ('datapush1', 'reupload')
            
        Returns:
            Dictionary containing iQID data and metadata
        """
        if sample_key not in self.sample_matches:
            logger.warning(f"Sample {sample_key} not found")
            return None
        
        iqid_matches = self.sample_matches[sample_key]['iqid']
        
        if location not in iqid_matches:
            logger.warning(f"Location {location} not available for {sample_key}")
            return None
        
        iqid_info = iqid_matches[location]
        data_types = iqid_info['data_types']
        
        if data_type not in data_types:
            logger.warning(f"Data type {data_type} not available for {sample_key} in {location}")
            return None
        
        try:
            data_files = data_types[data_type]
            
            if data_type == 'raw':
                # Single raw file
                return {
                    'raw_file': str(data_files),
                    'metadata': {
                        'sample_id': iqid_info['sample_id'],
                        'tissue_type': iqid_info['tissue_type'],
                        'scan_type': iqid_info['scan_type'],
                        'location': location,
                        'data_type': data_type
                    },
                    'path': str(iqid_info['path'])
                }
            else:
                # Multiple processed files
                file_paths = [str(f) for f in data_files]
                return {
                    'image_stack': file_paths,
                    'metadata': {
                        'sample_id': iqid_info['sample_id'],
                        'tissue_type': iqid_info['tissue_type'],
                        'scan_type': iqid_info['scan_type'],
                        'location': location,
                        'data_type': data_type,
                        'frame_count': len(file_paths)
                    },
                    'path': str(iqid_info['path'])
                }
                
        except Exception as e:
            logger.error(f"Failed to load iQID data for {sample_key}: {e}")
            return None
    
    def load_iqid_data_by_key(self, sample_key: str, preferred_location: str = None) -> Optional[Dict[str, Any]]:
        """
        Load iQID data by sample key (works for any iQID sample, not just matched pairs).
        
        Args:
            sample_key: iQID sample key (e.g., 'D7M2(P1)_L')
            preferred_location: Preferred data location ('datapush1' or 'reupload')
            
        Returns:
            Dictionary containing iQID data and metadata
        """
        if sample_key not in self.iqid_samples:
            logger.warning(f"iQID sample {sample_key} not found")
            return None
        
        iqid_info = self.iqid_samples[sample_key]
        
        # Choose location (prefer user preference, then reupload for raw data, then datapush1)
        location = preferred_location or 'reupload'
        if location not in iqid_info:
            location = list(iqid_info.keys())[0]  # Use first available
        
        location_info = iqid_info[location]
        
        # Choose data type (prefer raw, then aligned, then processed)
        data_types = location_info['data_types']
        if 'raw' in data_types:
            data_type = 'raw'
        elif 'aligned' in data_types:
            data_type = 'aligned'
        else:
            data_type = list(data_types.keys())[0]
        
        data_files = data_types[data_type]
        
        try:
            if len(data_files) == 1:
                # Single file (usually raw)
                return {
                    'raw_file': str(data_files[0]),
                    'metadata': {
                        'sample_id': location_info['sample_id'],
                        'tissue_type': location_info['tissue_type'],
                        'scan_type': location_info['scan_type'],
                        'location': location,
                        'data_type': data_type + '_only'  # Mark as iQID-only sample
                    },
                    'path': str(location_info['path'])
                }
            else:
                # Multiple processed files
                file_paths = [str(f) for f in data_files]
                return {
                    'image_stack': file_paths,
                    'metadata': {
                        'sample_id': location_info['sample_id'],
                        'tissue_type': location_info['tissue_type'],
                        'scan_type': location_info['scan_type'],
                        'location': location,
                        'data_type': data_type + '_only',  # Mark as iQID-only sample
                        'frame_count': len(file_paths)
                    },
                    'path': str(location_info['path'])
                }
                
        except Exception as e:
            logger.error(f"Failed to load iQID data for {sample_key}: {e}")
            return None

    def get_sample_summary(self) -> Dict[str, Any]:
        """Get summary of all available samples."""
        summary = {
            'total_matched_samples': len(self.sample_matches),
            'samples_by_tissue': {'kidney': 0, 'tumor': 0},
            'samples_by_side': {'L': 0, 'R': 0},
            'available_iqid_locations': set(),
            'sample_details': []
        }
        
        for sample_key, sample_info in self.sample_matches.items():
            he_info = sample_info['he']
            iqid_info = sample_info['iqid']
            
            # Count by tissue type
            tissue = he_info['tissue_type']
            summary['samples_by_tissue'][tissue] += 1
            
            # Count by side
            side = he_info['side']
            summary['samples_by_side'][side] += 1
            
            # Track iQID locations
            for location in iqid_info.keys():
                summary['available_iqid_locations'].add(location)
            
            # Sample details
            sample_detail = {
                'sample_key': sample_key,
                'he_files': he_info['file_count'],
                'tissue_type': tissue,
                'side': side,
                'iqid_locations': list(iqid_info.keys()),
                'iqid_data_types': {}
            }
            
            for location, iqid_data in iqid_info.items():
                sample_detail['iqid_data_types'][location] = list(iqid_data['data_types'].keys())
            
            summary['sample_details'].append(sample_detail)
        
        summary['available_iqid_locations'] = list(summary['available_iqid_locations'])
        
        return summary
    
    def get_unmatched_samples(self) -> Dict[str, Any]:
        """Get information about samples that couldn't be matched."""
        # Re-scan to get all samples
        he_samples = self._find_he_samples()
        iqid_samples = self._find_iqid_samples()
        
        unmatched_info = {
            'he_only': [],
            'iqid_only': [],
            'summary': {
                'total_he': len(he_samples),
                'total_iqid': len(iqid_samples),
                'matched': len(self.sample_matches),
                'he_unmatched': 0,
                'iqid_unmatched': 0
            }
        }
        
        # Find H&E samples without iQID matches
        for he_key, he_info in he_samples.items():
            if he_key not in self.sample_matches:
                unmatched_info['he_only'].append({
                    'sample_key': he_key,
                    'base_id': he_info['base_id'],
                    'side': he_info['side'],
                    'tissue_type': he_info['tissue_type'],
                    'scan_type': he_info['scan_type'],
                    'file_count': he_info['file_count']
                })
        
        # Find iQID samples without H&E matches
        matched_iqid_base_ids = set()
        for match_info in self.sample_matches.values():
            he_base_id = match_info['he']['base_id']
            he_side = match_info['he']['side']
            matched_iqid_base_ids.add(f"{he_base_id}_{he_side}")
        
        for iqid_key, iqid_info in iqid_samples.items():
            iqid_match_key = f"{iqid_info['base_id']}_{iqid_info['side']}"
            if iqid_match_key not in matched_iqid_base_ids:
                unmatched_info['iqid_only'].append({
                    'sample_id': iqid_info['sample_id'],
                    'base_id': iqid_info['base_id'],
                    'side': iqid_info['side'],
                    'tissue_type': iqid_info['tissue_type'],
                    'scan_type': iqid_info['scan_type'],
                    'location': iqid_info['location']
                })
        
        unmatched_info['summary']['he_unmatched'] = len(unmatched_info['he_only'])
        unmatched_info['summary']['iqid_unmatched'] = len(unmatched_info['iqid_only'])
        
        return unmatched_info
    
    def get_extended_summary(self) -> Dict[str, Any]:
        """Get extended summary including unmatched samples."""
        basic_summary = self.get_sample_summary()
        unmatched_info = self.get_unmatched_samples()
        
        extended_summary = {
            **basic_summary,
            'unmatched_samples': unmatched_info,
            'matching_efficiency': {
                'he_match_rate': (basic_summary['total_matched_samples'] / unmatched_info['summary']['total_he']) * 100 if unmatched_info['summary']['total_he'] > 0 else 0,
                'iqid_utilization_rate': (basic_summary['total_matched_samples'] / unmatched_info['summary']['total_iqid']) * 100 if unmatched_info['summary']['total_iqid'] > 0 else 0
            }
        }
        
        return extended_summary
