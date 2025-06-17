#!/usr/bin/env python3
"""
Detailed debug of sample matching logic
"""

import os
import json
from pathlib import Path
import re

def debug_sample_matching():
    """Debug the exact sample matching process."""
    print("🔍 Detailed Sample Matching Debug")
    print("=" * 50)
    
    config_path = "configs/unified_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_path = Path(config.get("data_paths", {}).get("base_path", ""))
    
    # Simulate the data loader logic exactly
    he_samples = {}
    iqid_samples = {}
    
    print("📊 H&E Sample Discovery:")
    print("-" * 30)
    
    # Find H&E samples (exactly like the data loader)
    he_base = base_path / "DataPush1" / "HE"
    
    for scan_type in ["3D", "Sequential sections (10um)"]:
        scan_path = he_base / scan_type
        if not scan_path.exists():
            continue
            
        print(f"  Scanning: {scan_path}")
            
        for tissue_type in ["kidney", "tumor"]:
            tissue_path = scan_path / tissue_type
            if not tissue_path.exists():
                continue
            
            print(f"    Tissue: {tissue_path}")
            
            # Find sample directories
            for sample_dir in tissue_path.iterdir():
                if not sample_dir.is_dir():
                    continue
                
                sample_id = sample_dir.name
                # Parse sample ID (e.g., D1M1_L, D7M2_R)
                match = re.match(r'(D\d+M\d+)_([LR])', sample_id)
                if not match:
                    print(f"      ❌ {sample_id} - No pattern match")
                    continue
                
                base_id, side = match.groups()
                image_files = list(sample_dir.glob("P*.tif"))
                
                key = f"{base_id}_{side}"
                he_samples[key] = {
                    'sample_id': sample_id,
                    'base_id': base_id,
                    'side': side,
                    'scan_type': scan_type,
                    'tissue_type': tissue_type,
                    'path': sample_dir,
                    'file_count': len(image_files)
                }
                
                print(f"      ✅ {sample_id} -> key: {key}, base_id: {base_id}, side: {side}")
    
    print(f"\n📊 iQID Sample Discovery:")
    print("-" * 30)
    
    # Find iQID samples (exactly like the data loader)
    locations = [
        ("DataPush1", "DataPush1/iQID", ["3D", "Sequential sections"]),
        ("ReUpload", "ReUpload/iQID_reupload/iQID", ["3D", "Sequential scans"])
    ]
    
    for location_name, location_path, scan_types in locations:
        iqid_base = base_path / location_path
        if not iqid_base.exists():
            continue
        
        print(f"  Location: {location_name} ({iqid_base})")
        
        for scan_type in scan_types:
            scan_path = iqid_base / scan_type
            if not scan_path.exists():
                continue
                
            print(f"    Scan type: {scan_type}")
                
            for tissue_type in ["kidney", "tumor"]:
                tissue_path = scan_path / tissue_type
                if not tissue_path.exists():
                    continue
                
                print(f"      Tissue: {tissue_type}")
                
                # Find sample directories
                for sample_dir in tissue_path.iterdir():
                    if not sample_dir.is_dir():
                        continue
                    
                    sample_id = sample_dir.name
                    # Parse iQID sample ID (e.g., D1M1(P1)_L, D7M2(P2)_R)
                    match = re.match(r'(D\d+M\d+)(?:\(P\d+\))?_([LR])', sample_id)
                    if not match:
                        print(f"        ❌ {sample_id} - No pattern match")
                        continue
                    
                    base_id, side = match.groups()
                    
                    key = f"{location_name.lower()}_{scan_type}_{tissue_type}_{base_id}_{side}_{sample_id}"
                    iqid_samples[key] = {
                        'sample_id': sample_id,
                        'base_id': base_id,
                        'side': side,
                        'location': location_name,
                        'scan_type': scan_type,
                        'tissue_type': tissue_type,
                        'path': sample_dir
                    }
                    
                    print(f"        ✅ {sample_id} -> base_id: {base_id}, side: {side}")
    
    print(f"\n📊 Matching Process:")
    print("-" * 30)
    
    matches = {}
    
    print(f"H&E samples found: {len(he_samples)}")
    for key, info in he_samples.items():
        print(f"  {key}: {info['base_id']}_{info['side']} ({info['tissue_type']})")
    
    print(f"\niQID samples found: {len(iqid_samples)}")
    for key, info in iqid_samples.items():
        print(f"  {info['sample_id']}: {info['base_id']}_{info['side']} ({info['tissue_type']}, {info['location']})")
    
    print(f"\nMatching H&E to iQID:")
    for he_key, he_info in he_samples.items():
        base_id = he_info['base_id']
        side = he_info['side']
        
        print(f"\n  Looking for matches for H&E {he_key} (base_id: {base_id}, side: {side}):")
        
        iqid_matches = {}
        
        for iqid_key, iqid_info in iqid_samples.items():
            if (iqid_info['base_id'] == base_id and 
                iqid_info['side'] == side):
                
                location = iqid_info['location']
                if location.lower() not in iqid_matches:
                    iqid_matches[location.lower()] = []
                iqid_matches[location.lower()].append(iqid_info)
                
                print(f"    ✅ Match: {iqid_info['sample_id']} in {location}")
        
        if iqid_matches:
            matches[he_key] = {
                'he': he_info,
                'iqid': iqid_matches
            }
            print(f"    Final match: {he_key} -> {len(iqid_matches)} iQID location(s)")
        else:
            print(f"    ❌ No iQID matches found")
    
    print(f"\n📊 Final Results:")
    print(f"Total matches: {len(matches)}")
    for match_key, match_info in matches.items():
        he_info = match_info['he']
        iqid_locations = list(match_info['iqid'].keys())
        print(f"  {match_key}: {he_info['tissue_type']} - iQID in {', '.join(iqid_locations)}")

if __name__ == "__main__":
    debug_sample_matching()
