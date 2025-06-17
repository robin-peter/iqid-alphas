#!/usr/bin/env python3
"""
Debug script to examine UCSF data structure and understand sample discovery.
"""

import os
import json
from pathlib import Path
import re

def explore_directory_structure():
    """Explore the UCSF data directory structure in detail."""
    print("🔍 UCSF Data Structure Analysis")
    print("=" * 60)
    
    # Load config to get base path
    config_path = "configs/unified_config.json"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_path = Path(config.get("data_paths", {}).get("base_path", ""))
    
    if not base_path.exists():
        print(f"❌ Base path not found: {base_path}")
        return
    
    print(f"📁 Base path: {base_path}")
    
    # Explore H&E data structure
    print(f"\n📊 H&E Data Structure:")
    he_base = base_path / "DataPush1" / "HE"
    
    if he_base.exists():
        for scan_type in he_base.iterdir():
            if scan_type.is_dir():
                print(f"  📁 {scan_type.name}/")
                
                for tissue_type in scan_type.iterdir():
                    if tissue_type.is_dir():
                        print(f"    📁 {tissue_type.name}/")
                        
                        sample_count = 0
                        for sample_dir in tissue_type.iterdir():
                            if sample_dir.is_dir():
                                sample_count += 1
                                if sample_count <= 5:  # Show first 5 samples
                                    image_files = list(sample_dir.glob("*.tif")) + list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
                                    print(f"      📁 {sample_dir.name}/ ({len(image_files)} files)")
                                    
                                    # Check if it matches our pattern
                                    match = re.match(r'(D\d+M\d+)_([LR])', sample_dir.name)
                                    pattern_match = "✅ Matches pattern" if match else "❌ No pattern match"
                                    print(f"         {pattern_match}")
                        
                        if sample_count > 5:
                            print(f"      ... and {sample_count - 5} more samples")
                        print(f"    Total {tissue_type.name} samples: {sample_count}")
    else:
        print(f"  ❌ H&E base path not found: {he_base}")
    
    # Explore iQID data structure  
    print(f"\n📊 iQID Data Structure:")
    
    # DataPush1 iQID
    iqid_datapush1 = base_path / "DataPush1" / "iQID"
    if iqid_datapush1.exists():
        print(f"  📁 DataPush1/iQID/")
        for scan_type in iqid_datapush1.iterdir():
            if scan_type.is_dir():
                print(f"    📁 {scan_type.name}/")
                
                for tissue_type in scan_type.iterdir():
                    if tissue_type.is_dir():
                        print(f"      📁 {tissue_type.name}/")
                        
                        sample_count = 0
                        for sample_dir in tissue_type.iterdir():
                            if sample_dir.is_dir():
                                sample_count += 1
                                if sample_count <= 5:
                                    data_files = list(sample_dir.glob("*.tif*"))
                                    print(f"        📁 {sample_dir.name}/ ({len(data_files)} files)")
                                    
                                    # Check pattern match
                                    match = re.match(r'(D\d+M\d+)(?:\(P\d+\))?_([LR])', sample_dir.name)
                                    pattern_match = "✅ Matches pattern" if match else "❌ No pattern match"
                                    print(f"           {pattern_match}")
                        
                        if sample_count > 5:
                            print(f"        ... and {sample_count - 5} more samples")
                        print(f"      Total {tissue_type.name} samples: {sample_count}")
    else:
        print(f"  ❌ DataPush1 iQID path not found: {iqid_datapush1}")
    
    # ReUpload iQID
    iqid_reupload = base_path / "ReUpload" / "iQID_reupload" / "iQID"
    if iqid_reupload.exists():
        print(f"  📁 ReUpload/iQID_reupload/iQID/")
        for scan_type in iqid_reupload.iterdir():
            if scan_type.is_dir():
                print(f"    📁 {scan_type.name}/")
                
                for tissue_type in scan_type.iterdir():
                    if tissue_type.is_dir():
                        print(f"      📁 {tissue_type.name}/")
                        
                        sample_count = 0
                        for sample_dir in tissue_type.iterdir():
                            if sample_dir.is_dir():
                                sample_count += 1
                                if sample_count <= 5:
                                    data_files = list(sample_dir.glob("*.tif"))
                                    print(f"        📁 {sample_dir.name}/ ({len(data_files)} files)")
                                    
                                    # Check pattern match
                                    match = re.match(r'(D\d+M\d+)(?:\(P\d+\))?_([LR])', sample_dir.name)
                                    pattern_match = "✅ Matches pattern" if match else "❌ No pattern match"
                                    print(f"           {pattern_match}")
                        
                        if sample_count > 5:
                            print(f"        ... and {sample_count - 5} more samples")
                        print(f"      Total {tissue_type.name} samples: {sample_count}")
    else:
        print(f"  ❌ ReUpload iQID path not found: {iqid_reupload}")

def analyze_sample_patterns():
    """Analyze all sample directory names to understand naming patterns."""
    print(f"\n🔍 Sample Pattern Analysis:")
    print("=" * 40)
    
    config_path = "configs/unified_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_path = Path(config.get("data_paths", {}).get("base_path", ""))
    
    all_sample_names = set()
    
    # Collect all sample directory names
    locations = [
        base_path / "DataPush1" / "HE" / "3D" / "kidney",
        base_path / "DataPush1" / "HE" / "3D" / "tumor", 
        base_path / "DataPush1" / "HE" / "Sequential sections (10um)" / "kidney",
        base_path / "DataPush1" / "HE" / "Sequential sections (10um)" / "tumor",
        base_path / "DataPush1" / "iQID" / "3D" / "kidney",
        base_path / "DataPush1" / "iQID" / "3D" / "tumor",
        base_path / "DataPush1" / "iQID" / "Sequential sections" / "kidney",
        base_path / "DataPush1" / "iQID" / "Sequential sections" / "tumor",
        base_path / "ReUpload" / "iQID_reupload" / "iQID" / "3D" / "kidney",
        base_path / "ReUpload" / "iQID_reupload" / "iQID" / "3D" / "tumor",
        base_path / "ReUpload" / "iQID_reupload" / "iQID" / "Sequential scans" / "kidney",
        base_path / "ReUpload" / "iQID_reupload" / "iQID" / "Sequential scans" / "tumor"
    ]
    
    for location in locations:
        if location.exists():
            for sample_dir in location.iterdir():
                if sample_dir.is_dir():
                    all_sample_names.add(sample_dir.name)
    
    print(f"📊 Found {len(all_sample_names)} unique sample directory names:")
    
    # Categorize by pattern
    he_pattern = re.compile(r'(D\d+M\d+)_([LR])$')
    iqid_pattern = re.compile(r'(D\d+M\d+)(?:\(P\d+\))?_([LR])$')
    other_patterns = []
    
    he_samples = []
    iqid_samples = []
    
    for name in sorted(all_sample_names):
        if he_pattern.match(name):
            he_samples.append(name)
        elif iqid_pattern.match(name):
            iqid_samples.append(name)
        else:
            other_patterns.append(name)
    
    print(f"\n✅ H&E Pattern Matches (D#M#_[LR]):")
    for sample in he_samples:
        print(f"   {sample}")
    
    print(f"\n✅ iQID Pattern Matches (D#M#(P#)?_[LR]):")
    for sample in iqid_samples:
        print(f"   {sample}")
    
    if other_patterns:
        print(f"\n❓ Other patterns found:")
        for sample in other_patterns[:10]:  # Show first 10
            print(f"   {sample}")
        if len(other_patterns) > 10:
            print(f"   ... and {len(other_patterns) - 10} more")
    
    print(f"\n📊 Summary:")
    print(f"   H&E pattern matches: {len(he_samples)}")
    print(f"   iQID pattern matches: {len(iqid_samples)}")
    print(f"   Other patterns: {len(other_patterns)}")

if __name__ == "__main__":
    explore_directory_structure()
    analyze_sample_patterns()
