#!/usr/bin/env python3
"""
IQID-Alphas Command Line Interface

Advanced batch processing CLI for the IQID-Alphas project providing:
- Unified interface for all processing workflows
- Batch processing with progress monitoring
- Quality control and validation
- Automated report generation
- Configuration management
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import traceback

# Add package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from iqid_alphas.pipelines.simple import SimplePipeline
from iqid_alphas.pipelines.advanced import AdvancedPipeline
from iqid_alphas.pipelines.combined import CombinedPipeline


class IQIDCLIProcessor:
    """Advanced CLI processor for IQID-Alphas batch operations."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.start_time = datetime.now()
        self.results = {
            'processed': 0,
            'success': 0,
            'failed': 0,
            'errors': [],
            'summary': {}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('iqid_cli')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _detect_dataset_type(self, data_path: Path) -> str:
        """Detect if this is DataPush1 (production) or ReUpload (workflow) dataset."""
        path_str = str(data_path).lower()
        
        if "datapush1" in path_str:
            return "production"  # DataPush1: aligned data with H&E
        elif "reupload" in path_str or "iqid_reupload" in path_str:
            return "workflow"    # ReUpload: full workflow, iQID only
        else:
            # Try to detect by structure
            if (data_path / "HE").exists():
                return "production"  # Has H&E data = production dataset
            elif (data_path / "Raw").exists() or any(p.name == "Raw" for p in data_path.rglob("Raw")):
                return "workflow"    # Has Raw directories = workflow dataset
            else:
                return "unknown"
    
    def _detect_processing_stage(self, sample_dir: Path, dataset_type: str) -> str:
        """Detect processing stage based on dataset type and available directories."""
        if dataset_type == "production":
            # DataPush1: Only aligned data available
            return "aligned_ready"
        elif dataset_type == "workflow":
            # ReUpload: Check for workflow stages
            if (sample_dir / "Raw").exists():
                return "raw_available"
            elif (sample_dir / "1_segmented").exists():
                return "segmented_available"
            elif (sample_dir / "2_aligned").exists():
                return "aligned_available"
            elif any(sample_dir.glob("mBq_corr_*.tif")):
                # Direct slice files (already processed)
                return "slices_available"
        return "unknown_stage"
    
    def discover_data(self, data_path: Union[str, Path]) -> Dict[str, List[Path]]:
        """Discover available data samples with dataset-aware processing."""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Detect dataset type
        dataset_type = self._detect_dataset_type(data_path)
        
        discovered = {
            'dataset_type': dataset_type,
            'iqid_samples': [],
            'he_samples': [],
            'paired_samples': [],
            'sample_metadata': {},
            'dataset_info': {
                'type': dataset_type,
                'has_he_data': dataset_type == "production",
                'has_workflow_stages': dataset_type == "workflow",
                'multi_modal_capable': dataset_type == "production"
            }
        }
        
        self.logger.info(f"Detected dataset type: {dataset_type}")
        
        # Dataset-specific discovery
        if dataset_type == "production":
            self._discover_production_samples(data_path, discovered)
        elif dataset_type == "workflow":
            self._discover_workflow_samples(data_path, discovered)
        else:
            self._discover_generic_samples(data_path, discovered)
        
        return discovered
    
    def _analyze_sample_directory(self, sample_dir: Path, data_type: str) -> Dict:
        """Analyze a sample directory and detect iQID processing stage (Raw/Segmented/Aligned)."""
        
        # Detect iQID processing stages
        if data_type == 'iqid':
            processing_stage, files = self._detect_iqid_processing_stage(sample_dir)
        elif data_type == 'he':
            files = list(sample_dir.glob('P*.tif')) + list(sample_dir.glob('*-T*-P*.tif'))
            processing_stage = 'he_slices'
        else:
            files = []
            processing_stage = 'unknown'
        
        # Sort files by slice number for proper ordering
        if files and data_type == 'iqid' and processing_stage in ['segmented', 'aligned']:
            sorted_files = self._sort_slice_files(files)
        else:
            sorted_files = sorted(files)
        
        # Extract tissue type and preprocessing type from path
        path_parts = sample_dir.parts
        tissue_type = None
        preprocessing_type = None
        
        for part in path_parts:
            if part.lower() in ['kidney', 'tumor']:
                tissue_type = part.lower()
            elif part.lower() in ['3d', 'sequential', 'upper', 'lower']:
                preprocessing_type = part.lower()
        
        return {
            'sample_dir': sample_dir,
            'sample_id': sample_dir.name,
            'data_type': data_type,
            'tissue_type': tissue_type,
            'preprocessing_type': preprocessing_type,
            'processing_stage': processing_stage,
            'slice_count': len(files),
            'slice_files': sorted_files,
            'size_mb': sum(f.stat().st_size for f in files) / (1024 * 1024) if files else 0,
            'can_reconstruct_3d': processing_stage == 'aligned' and len(files) > 1,
            'workflow_ready': self._assess_workflow_readiness(processing_stage, len(files))
        }
    
    def _pair_samples(self, iqid_samples: List[Dict], he_samples: List[Dict]) -> List[Dict]:
        """Attempt to pair iQID samples with H&E samples based on naming patterns."""
        paired = []
        
        for iqid_sample in iqid_samples:
            # Extract potential sample identifier from directory name
            iqid_id = self._extract_sample_id(iqid_sample['sample_id'])
            
            best_match = None
            best_score = 0
            
            for he_sample in he_samples:
                he_id = self._extract_sample_id(he_sample['sample_id'])
                score = self._similarity_score(iqid_id, he_id)
                
                if score > best_score and score > 0.3:  # Threshold for matching
                    best_match = he_sample
                    best_score = score
            
            paired.append({
                'iqid_sample': iqid_sample,
                'he_sample': best_match,
                'sample_id': iqid_id,
                'match_score': best_score,
                'type': 'sample_pair'  # Indicates this is a sample pair, not individual image
            })
        
        return paired
    
    def _extract_sample_id(self, filename: str) -> str:
        """Extract sample identifier from filename or directory name."""
        # Handle directory names like D1M1(P1)_L, D7M2(P2)_R, etc.
        base = filename
        
        # Remove common file extensions
        if '.' in base:
            base = Path(base).stem
        
        # Handle iQID naming pattern: D1M1(P1)_L -> D1M1_L
        # Remove (P*) parts to match with H&E names
        import re
        base = re.sub(r'\(P\d+\)', '', base)
        
        # Clean up any double underscores
        base = re.sub(r'_+', '_', base)
        
        # Remove common prefixes/suffixes
        patterns_to_remove = [
            'mBq_corr_', 'aligned_', 'processed_', 'segmented_',
            '_he', '_iqid', '_corr', '_aligned'
        ]
        
        for pattern in patterns_to_remove:
            base = base.replace(pattern, '')
        
        return base.lower().strip('_')
    
    def _similarity_score(self, str1: str, str2: str) -> float:
        """Calculate similarity score between two strings."""
        if not str1 or not str2:
            return 0.0
        
        # Simple Jaccard similarity based on character sets
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def process_batch(self, config_path: str, data_path: str, 
                     pipeline_type: str = 'simple', max_samples: Optional[int] = None,
                     output_dir: Optional[str] = None) -> Dict:
        """Process a batch of samples using the specified pipeline."""
        
        self.logger.info(f"🚀 Starting batch processing with {pipeline_type} pipeline")
        self.logger.info(f"Data path: {data_path}")
        self.logger.info(f"Config: {config_path}")
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return self.results
        
        # Discover data
        try:
            discovered = self.discover_data(data_path)
        except Exception as e:
            self.logger.error(f"Data discovery failed: {e}")
            return self.results
        
        # Select pipeline
        pipeline_classes = {
            'simple': SimplePipeline,
            'advanced': AdvancedPipeline,
            'combined': CombinedPipeline
        }
        
        if pipeline_type not in pipeline_classes:
            self.logger.error(f"Unknown pipeline type: {pipeline_type}")
            return self.results
        
        PipelineClass = pipeline_classes[pipeline_type]
        
        # Process samples
        samples_to_process = discovered['paired_samples']
        if max_samples:
            samples_to_process = samples_to_process[:max_samples]
        
        self.logger.info(f"Processing {len(samples_to_process)} samples...")
        
        for i, sample in enumerate(samples_to_process, 1):
            try:
                self.logger.info(f"Processing sample {i}/{len(samples_to_process)}: {sample['sample_id']}")
                
                # Create pipeline instance
                pipeline = PipelineClass(config)
                
                # Set up output directory for this sample
                if output_dir:
                    sample_output_dir = Path(output_dir) / f"sample_{sample['sample_id']}"
                    sample_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Update config with sample-specific output directory
                    sample_config = config.copy()
                    if 'visualization' not in sample_config:
                        sample_config['visualization'] = {}
                    sample_config['visualization']['output_dir'] = str(sample_output_dir)
                    
                    pipeline = PipelineClass(sample_config)
                
                # Process based on pipeline type and sample structure
                iqid_sample = sample['iqid_sample']
                he_sample = sample.get('he_sample')
                
                if pipeline_type == 'simple':
                    # SimplePipeline processes the entire slice stack for 3D reconstruction
                    sample_output_dir = sample_output_dir if output_dir else "results"
                    result = pipeline.process_iqid_stack(str(iqid_sample['sample_dir']), str(sample_output_dir))
                elif pipeline_type == 'advanced':
                    # AdvancedPipeline - process middle slice as representative or whole stack
                    sample_output_dir = sample_output_dir if output_dir else "results"
                    if iqid_sample['slice_files']:
                        # Use middle slice as most representative
                        mid_idx = len(iqid_sample['slice_files']) // 2
                        representative_slice = iqid_sample['slice_files'][mid_idx]
                        result = pipeline.process_image(str(representative_slice), str(sample_output_dir))
                        
                        # Log slice information
                        self.logger.info(f"Processing slice {mid_idx+1}/{len(iqid_sample['slice_files'])} as representative")
                    else:
                        raise ValueError(f"No slices found in sample {iqid_sample['sample_id']}")
                elif pipeline_type == 'combined' and he_sample:
                    # CombinedPipeline uses corresponding slices from both modalities
                    sample_output_dir = sample_output_dir if output_dir else "results"
                    if iqid_sample['slice_files'] and he_sample['slice_files']:
                        # Use corresponding slices (same index) or middle slices
                        iqid_mid = len(iqid_sample['slice_files']) // 2
                        he_mid = len(he_sample['slice_files']) // 2
                        
                        iqid_slice = iqid_sample['slice_files'][iqid_mid]
                        he_slice = he_sample['slice_files'][he_mid]
                        
                        result = pipeline.process_image_pair(
                            str(he_slice), 
                            str(iqid_slice),
                            str(sample_output_dir)
                        )
                        
                        self.logger.info(f"Processing iQID slice {iqid_mid+1}/{len(iqid_sample['slice_files'])} "
                                       f"with H&E slice {he_mid+1}/{len(he_sample['slice_files'])}")
                    else:
                        raise ValueError(f"Missing slices in sample pair {sample['sample_id']}")
                else:
                    # Fallback: process middle slice from iQID stack
                    sample_output_dir = sample_output_dir if output_dir else "results"
                    if iqid_sample['slice_files']:
                        mid_idx = len(iqid_sample['slice_files']) // 2
                        representative_slice = iqid_sample['slice_files'][mid_idx]
                        result = pipeline.process_image(str(representative_slice), str(sample_output_dir))
                    else:
                        raise ValueError(f"No slices found in sample {iqid_sample['sample_id']}")
                
                self.results['processed'] += 1
                self.results['success'] += 1
                
                self.logger.info(f"✅ Successfully processed {sample['sample_id']}")
                
            except Exception as e:
                self.results['processed'] += 1
                self.results['failed'] += 1
                error_msg = f"Failed to process {sample['sample_id']}: {str(e)}"
                self.results['errors'].append(error_msg)
                self.logger.error(f"❌ {error_msg}")
                self.logger.debug(traceback.format_exc())
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate processing summary."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.results['summary'] = {
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'success_rate': (self.results['success'] / self.results['processed'] * 100) 
                           if self.results['processed'] > 0 else 0
        }
        
        self.logger.info("\n" + "="*50)
        self.logger.info("📊 BATCH PROCESSING SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total processed: {self.results['processed']}")
        self.logger.info(f"Successful: {self.results['success']}")
        self.logger.info(f"Failed: {self.results['failed']}")
        self.logger.info(f"Success rate: {self.results['summary']['success_rate']:.1f}%")
        self.logger.info(f"Duration: {duration.total_seconds():.1f} seconds")
        
        if self.results['errors']:
            self.logger.info(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                self.logger.info(f"  - {error}")
    
    def _analyze_sample_directory(self, sample_dir: Path, image_type: str) -> Dict:
        """Analyze a sample directory to extract metadata."""
        sample_info = {
            'sample_path': sample_dir,
            'sample_name': sample_dir.name,
            'image_type': image_type,
            'tissue_type': self._extract_tissue_type(sample_dir),
            'preprocessing_type': self._extract_preprocessing_type(sample_dir),
            'laterality': self._extract_laterality(sample_dir.name),
            'time_points': [],
            'file_count': 0
        }
        
        # Count and catalog time points
        if image_type == 'iqid':
            time_files = sorted(sample_dir.glob('mBq_corr_*.tif'), 
                              key=lambda x: int(x.stem.split('_')[-1]))
            sample_info['time_points'] = [f.name for f in time_files]
            sample_info['file_count'] = len(time_files)
        elif image_type == 'he':
            time_files = sorted(sample_dir.glob('P*.tif'))
            if not time_files:  # Try tumor naming pattern
                time_files = sorted(sample_dir.glob('*-T*-P*.tif'))
            sample_info['time_points'] = [f.name for f in time_files]
            sample_info['file_count'] = len(time_files)
        
        return sample_info
    
    def _extract_tissue_type(self, sample_path: Path) -> str:
        """Extract tissue type from path hierarchy."""
        path_parts = sample_path.parts
        for part in path_parts:
            if part.lower() in ['kidney', 'tumor', 'kidneys']:
                return part.lower()
        return 'unknown'
    
    def _extract_preprocessing_type(self, sample_path: Path) -> str:
        """Extract preprocessing type from path hierarchy."""
        path_parts = sample_path.parts
        for part in path_parts:
            if part.lower() in ['3d', 'sequential', 'upper and lower']:
                return part.lower()
        return 'unknown'
    
    def _extract_laterality(self, sample_name: str) -> str:
        """Extract laterality (left/right) from sample name."""
        if sample_name.endswith('_L') or 'Left' in sample_name:
            return 'left'
        elif sample_name.endswith('_R') or 'Right' in sample_name:
            return 'right'
        return 'unknown'


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='IQID-Alphas Advanced Batch Processing CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic batch processing
  python -m iqid_alphas.cli process --data /path/to/data --config configs/simple.json
  
  # Advanced pipeline with custom output
  python -m iqid_alphas.cli process --data /path/to/data --config configs/advanced.json \\
                                   --pipeline advanced --output results/batch_1
  
  # Quick test with limited samples
  python -m iqid_alphas.cli process --data /path/to/data --config configs/test.json \\
                                   --max-samples 5 --verbose
  
  # Discover available data
  python -m iqid_alphas.cli discover --data /path/to/data
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process batch of samples')
    process_parser.add_argument('--data', required=True, 
                              help='Path to data directory')
    process_parser.add_argument('--config', required=True,
                              help='Path to configuration file')
    process_parser.add_argument('--pipeline', choices=['simple', 'advanced', 'combined'],
                              default='simple', help='Pipeline type to use')
    process_parser.add_argument('--output', help='Output directory for results')
    process_parser.add_argument('--max-samples', type=int,
                              help='Maximum number of samples to process')
    process_parser.add_argument('--verbose', action='store_true',
                              help='Enable verbose logging')
    
    # Discover command
    discover_parser = subparsers.add_parser('discover', help='Discover available data files')
    discover_parser.add_argument('--data', required=True,
                               help='Path to data directory')
    discover_parser.add_argument('--output', help='Save discovery results to JSON file')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--create', help='Create default configuration file')
    config_parser.add_argument('--validate', help='Validate configuration file')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    cli_processor = IQIDCLIProcessor()
    
    try:
        if args.command == 'process':
            results = cli_processor.process_batch(
                config_path=args.config,
                data_path=args.data,
                pipeline_type=args.pipeline,
                max_samples=args.max_samples,
                output_dir=args.output
            )
            
            # Save results if output directory specified
            if args.output:
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)
                results_file = output_path / 'batch_results.json'
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                cli_processor.logger.info(f"Results saved to: {results_file}")
        
        elif args.command == 'discover':
            discovered = cli_processor.discover_data(args.data)
            
            print("\n📁 DATA DISCOVERY RESULTS")
            print("="*40)
            print(f"iQID samples (tissue slice stacks): {len(discovered['iqid_samples'])}")
            print(f"H&E samples (tissue slice stacks): {len(discovered['he_samples'])}")
            print(f"Paired samples (multi-modal): {len(discovered['paired_samples'])}")
            
            # Show detailed breakdown
            if discovered['iqid_samples']:
                print(f"\n📊 Sample Details:")
                tissue_types = {}
                total_slices = 0
                samples_3d = 0
                
                for sample in discovered['iqid_samples']:
                    tissue = sample.get('tissue_type', 'unknown')
                    tissue_types[tissue] = tissue_types.get(tissue, 0) + 1
                    total_slices += sample.get('slice_count', 0)
                    if sample.get('can_reconstruct_3d', False):
                        samples_3d += 1
                
                for tissue, count in tissue_types.items():
                    print(f"   - {tissue}: {count} samples")
                
                print(f"\n🧬 Slice Information:")
                print(f"   - Total tissue slices: {total_slices}")
                print(f"   - Samples suitable for 3D reconstruction: {samples_3d}")
                print(f"   - Average slices per sample: {total_slices/len(discovered['iqid_samples']):.1f}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(discovered, f, indent=2, default=str)
                print(f"\nResults saved to: {args.output}")
        
        elif args.command == 'config':
            if args.create:
                create_default_config(args.create)
            elif args.validate:
                validate_config(args.validate)
    
    except KeyboardInterrupt:
        cli_processor.logger.info("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        cli_processor.logger.error(f"❌ CLI Error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            cli_processor.logger.error(traceback.format_exc())
        sys.exit(1)


def create_default_config(config_path: str):
    """Create a default configuration file."""
    default_config = {
        "processing": {
            "normalize": True,
            "remove_outliers": True,
            "gaussian_filter": True,
            "filter_sigma": 1.0
        },
        "segmentation": {
            "method": "otsu",
            "min_size": 100,
            "remove_small_objects": True
        },
        "alignment": {
            "method": "phase_correlation",
            "max_shift": 50,
            "subpixel_precision": True
        },
        "visualization": {
            "save_plots": True,
            "output_dir": "results",
            "dpi": 300,
            "format": "png"
        },
        "quality_control": {
            "enable_validation": True,
            "min_alignment_score": 0.5,
            "max_processing_time": 300
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"✅ Default configuration created: {config_path}")


def validate_config(config_path: str):
    """Validate a configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Basic validation
        required_sections = ['processing', 'segmentation', 'alignment', 'visualization']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            print(f"❌ Missing required sections: {missing_sections}")
            return False
        
        print(f"✅ Configuration file is valid: {config_path}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        return False


if __name__ == '__main__':
    main()
