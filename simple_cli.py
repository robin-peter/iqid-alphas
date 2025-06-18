#!/usr/bin/env python3
"""
IQID-Alphas CLI - Simple Working Version

Basic CLI for discovering and processing UCSF iQID data with understanding of:
- DataPush1: Production dataset (aligned iQID + H&E)
- ReUpload: Development dataset (full workflow stages)
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List


def detect_dataset_type(data_path: Path) -> str:
    """Detect if this is DataPush1 (production) or ReUpload (workflow) dataset."""
    path_str = str(data_path).lower()
    
    if 'datapush1' in path_str:
        return 'production'
    elif 'reupload' in path_str or 'iqid_reupload' in path_str:
        return 'workflow'
    else:
        return 'unknown'


def discover_datapush1(data_path: Path) -> Dict:
    """Discover DataPush1 production data (aligned iQID + H&E)."""
    results = {
        'dataset_type': 'production',
        'iqid_samples': [],
        'he_samples': [],
        'paired_samples': []
    }
    
    # Look for iQID samples in DataPush1/iQID structure
    iqid_path = data_path / 'iQID'
    if iqid_path.exists():
        for sample_dir in iqid_path.rglob('*'):
            if sample_dir.is_dir():
                # Look for mBq_corr_*.tif files (aligned iQID data)
                iqid_files = list(sample_dir.glob('mBq_corr_*.tif'))
                if iqid_files:
                    results['iqid_samples'].append({
                        'sample_id': sample_dir.name,
                        'sample_dir': sample_dir,
                        'slice_count': len(iqid_files),
                        'files': sorted(iqid_files),
                        'stage': 'aligned_ready'
                    })
    
    # Look for H&E samples in DataPush1/HE structure
    he_path = data_path / 'HE'
    if he_path.exists():
        for sample_dir in he_path.rglob('*'):
            if sample_dir.is_dir():
                # Look for P*.tif files (H&E data)
                he_files = list(sample_dir.glob('P*.tif'))
                if he_files:
                    results['he_samples'].append({
                        'sample_id': sample_dir.name,
                        'sample_dir': sample_dir,
                        'slice_count': len(he_files),
                        'files': sorted(he_files)
                    })
    
    # Simple pairing by similar names
    for iqid_sample in results['iqid_samples']:
        iqid_id = iqid_sample['sample_id'].replace('(P1)', '').replace('(P2)', '')
        for he_sample in results['he_samples']:
            if iqid_id.startswith(he_sample['sample_id']):
                results['paired_samples'].append({
                    'iqid_sample': iqid_sample,
                    'he_sample': he_sample,
                    'sample_id': iqid_id
                })
                break
    
    return results


def discover_reupload(data_path: Path) -> Dict:
    """Discover ReUpload workflow data (Raw → Segmented → Aligned)."""
    results = {
        'dataset_type': 'workflow',
        'samples_by_stage': {
            'raw': [],
            'segmented': [],
            'aligned': []
        },
        'workflow_summary': {}
    }
    
    # Look for workflow stages
    for stage_name in ['Raw', '1_segmented', '2_aligned']:
        stage_path = data_path / 'iQID_reupload' / stage_name
        if stage_path.exists():
            stage_key = stage_name.lower().replace('1_', '').replace('2_', '')
            
            for sample_dir in stage_path.iterdir():
                if sample_dir.is_dir():
                    files = list(sample_dir.glob('*.tif'))
                    if files:
                        results['samples_by_stage'][stage_key].append({
                            'sample_id': sample_dir.name,
                            'sample_dir': sample_dir,
                            'stage': stage_key,
                            'file_count': len(files),
                            'files': sorted(files)
                        })
    
    return results


def discover_data(data_path: str) -> Dict:
    """Main discovery function."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    dataset_type = detect_dataset_type(data_path)
    
    if dataset_type == 'production':
        return discover_datapush1(data_path)
    elif dataset_type == 'workflow':
        return discover_reupload(data_path)
    else:
        # Try to auto-detect by looking for characteristic structures
        if (data_path / 'iQID').exists() and (data_path / 'HE').exists():
            return discover_datapush1(data_path)
        elif (data_path / 'iQID_reupload').exists():
            return discover_reupload(data_path)
        else:
            return {'dataset_type': 'unknown', 'error': 'Could not determine dataset type'}


def print_discovery_results(results: Dict):
    """Print formatted discovery results."""
    print("\n📁 DATA DISCOVERY RESULTS")
    print("=" * 50)
    
    dataset_type = results.get('dataset_type', 'unknown')
    
    if dataset_type == 'production':
        print(f"Dataset Type: Production (DataPush1)")
        print(f"iQID samples (aligned, 3D ready): {len(results['iqid_samples'])}")
        print(f"H&E samples (ready): {len(results['he_samples'])}")
        print(f"Paired samples (multi-modal): {len(results['paired_samples'])}")
        
        if results['iqid_samples']:
            total_slices = sum(s['slice_count'] for s in results['iqid_samples'])
            print(f"\n🧬 Sample Details:")
            print(f"   - Total iQID slices: {total_slices}")
            print(f"   - Average slices per sample: {total_slices/len(results['iqid_samples']):.1f}")
            print(f"   - All samples ready for 3D reconstruction")
            print(f"   - Multi-modal analysis available (iQID + H&E)")
            
            print(f"\n📋 Sample List (first 5):")
            for i, sample in enumerate(results['iqid_samples'][:5]):
                print(f"   {i+1}. {sample['sample_id']} ({sample['slice_count']} slices)")
    
    elif dataset_type == 'workflow':
        print(f"Dataset Type: Workflow Development (ReUpload)")
        print(f"H&E samples: 0 (not available in workflow dataset)")
        
        stages = results['samples_by_stage']
        print(f"\n🔬 Workflow Stage Analysis:")
        print(f"   - Raw stage: {len(stages['raw'])} samples")
        print(f"   - Segmented stage: {len(stages['segmented'])} samples")
        print(f"   - Aligned stage: {len(stages['aligned'])} samples")
        
        if stages['aligned']:
            print(f"   - Ready for 3D reconstruction: {len(stages['aligned'])} samples")
        
        print(f"\n🧬 Workflow Opportunities:")
        raw_only = len(stages['raw']) - len(stages['segmented'])
        seg_only = len(stages['segmented']) - len(stages['aligned'])
        if raw_only > 0:
            print(f"   - {raw_only} samples need segmentation")
        if seg_only > 0:
            print(f"   - {seg_only} samples need alignment")
        if stages['aligned']:
            print(f"   - {len(stages['aligned'])} samples ready for analysis")
    
    else:
        print(f"Dataset Type: Unknown")
        if 'error' in results:
            print(f"Error: {results['error']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='IQID-Alphas Simple CLI for Data Discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover DataPush1 production data
  python simple_cli.py discover --data data/DataPush1
  
  # Discover ReUpload workflow data
  python simple_cli.py discover --data data/ReUpload
  
  # Save results to file
  python simple_cli.py discover --data data/DataPush1 --output results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Discover command
    discover_parser = subparsers.add_parser('discover', help='Discover available data')
    discover_parser.add_argument('--data', required=True, help='Path to data directory')
    discover_parser.add_argument('--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'discover':
            print(f"🔍 Discovering data in: {args.data}")
            results = discover_data(args.data)
            
            print_discovery_results(results)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\n💾 Results saved to: {args.output}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
