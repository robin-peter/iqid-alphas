#!/usr/bin/env python3
"""
Simplified iQID-Only Processing Pipeline
Processes iQID data from ReUpload directory through complete pipeline
"""

import os
import sys
import json
import numpy as np
import glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tifffile import imread, imwrite

# Add current directory to path for imports
sys.path.append('.')
sys.path.append('./src')

class SimpleiQIDPipeline:
    """Simplified iQID processing pipeline for ReUpload data"""
    
    def __init__(self, config_file='configs/iqid_pipeline_config.json'):
        print("🔬 Initializing Simplified iQID Processing Pipeline...")
        
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"   ⚠️  Config file not found: {config_file}")
            print("   🔧 Using default configuration...")
            self.config = {
                "paths": {
                    "reupload_root": "../data/UCSF-Collab/data/ReUpload",
                    "output_root": "./outputs/iqid_only_processing"
                }
            }
        
        self.reupload_root = self.config['paths']['reupload_root']
        self.output_root = self.config['paths']['output_root']
        
        # Create output directory
        os.makedirs(self.output_root, exist_ok=True)
        
        print("   ✓ iQID Processing Pipeline initialized")
    
    def discover_samples(self, max_samples=5):
        """Discover available iQID samples in ReUpload directory"""
        
        print(f"🔍 Discovering iQID samples in ReUpload directory...")
        
        iqid_dir = os.path.join(self.reupload_root, "iQID_reupload/iQID/Sequential/kidneys")
        
        if not os.path.exists(iqid_dir):
            print(f"   ❌ iQID directory not found: {iqid_dir}")
            return []
        
        samples = []
        for sample_dir in os.listdir(iqid_dir):
            sample_path = os.path.join(iqid_dir, sample_dir)
            if os.path.isdir(sample_path):
                
                # Look for raw event image
                event_files = glob.glob(os.path.join(sample_path, "*event_image.tif"))
                
                # Look for segmented images
                segmented_dir = os.path.join(sample_path, "1_segmented")
                segmented_files = []
                if os.path.exists(segmented_dir):
                    segmented_files = glob.glob(os.path.join(segmented_dir, "*.tif"))
                
                # Look for aligned images
                aligned_dir = os.path.join(sample_path, "2_aligned")
                aligned_files = []
                if os.path.exists(aligned_dir):
                    aligned_files = glob.glob(os.path.join(aligned_dir, "*.tif"))
                
                sample_info = {
                    'sample_id': sample_dir,
                    'sample_path': sample_path,
                    'raw_event_files': event_files,
                    'segmented_files': segmented_files,
                    'aligned_files': aligned_files,
                    'processing_stages': {
                        'has_raw': len(event_files) > 0,
                        'has_segmented': len(segmented_files) > 0,
                        'has_aligned': len(aligned_files) > 0
                    }
                }
                
                samples.append(sample_info)
                
                if len(samples) >= max_samples:
                    break
        
        print(f"   ✓ Discovered {len(samples)} samples")
        for sample in samples:
            stages = sample['processing_stages']
            stage_info = f"Raw: {'✓' if stages['has_raw'] else '✗'}, " + \
                        f"Seg: {'✓' if stages['has_segmented'] else '✗'}, " + \
                        f"Align: {'✓' if stages['has_aligned'] else '✗'}"
            print(f"     - {sample['sample_id']}: {stage_info}")
        
        return samples
    
    def analyze_processing_workflow(self, sample_info):
        """Analyze the complete processing workflow for a sample"""
        
        print(f"\\n📊 Analyzing processing workflow: {sample_info['sample_id']}")
        
        analysis = {
            'sample_id': sample_info['sample_id'],
            'workflow_stages': {},
            'file_analysis': {},
            'processing_quality': {}
        }
        
        # Stage 1: Raw Event Image Analysis
        if sample_info['raw_event_files']:
            raw_file = sample_info['raw_event_files'][0]
            raw_image = imread(raw_file)
            
            analysis['workflow_stages']['raw_event'] = {
                'file': os.path.basename(raw_file),
                'shape': raw_image.shape,
                'dtype': str(raw_image.dtype),
                'value_range': [float(raw_image.min()), float(raw_image.max())],
                'mean_intensity': float(raw_image.mean())
            }
            
            print(f"   🔬 Raw Event: {raw_image.shape}, range: {raw_image.min():.1f}-{raw_image.max():.1f}")
        
        # Stage 2: Segmented Images Analysis
        if sample_info['segmented_files']:
            segmented_files = sorted(sample_info['segmented_files'])
            
            # Analyze a few representative segmented images
            seg_analysis = []
            for i, seg_file in enumerate(segmented_files[:5]):  # Analyze first 5
                seg_image = imread(seg_file)
                seg_info = {
                    'file': os.path.basename(seg_file),
                    'shape': seg_image.shape,
                    'value_range': [float(seg_image.min()), float(seg_image.max())],
                    'non_zero_pixels': int(np.count_nonzero(seg_image)),
                    'coverage_percentage': float(np.count_nonzero(seg_image) / seg_image.size * 100)
                }
                seg_analysis.append(seg_info)
            
            analysis['workflow_stages']['segmented'] = {
                'total_files': len(segmented_files),
                'analyzed_files': len(seg_analysis),
                'representative_analysis': seg_analysis,
                'average_coverage': np.mean([s['coverage_percentage'] for s in seg_analysis])
            }
            
            print(f"   ✂️  Segmented: {len(segmented_files)} files, avg coverage: {analysis['workflow_stages']['segmented']['average_coverage']:.1f}%")
        
        # Stage 3: Aligned Images Analysis
        if sample_info['aligned_files']:
            aligned_files = sorted(sample_info['aligned_files'])
            
            # Analyze alignment quality by comparing with segmented
            align_analysis = []
            for i, align_file in enumerate(aligned_files[:5]):
                align_image = imread(align_file)
                align_info = {
                    'file': os.path.basename(align_file),
                    'shape': align_image.shape,
                    'value_range': [float(align_image.min()), float(align_image.max())],
                    'non_zero_pixels': int(np.count_nonzero(align_image)),
                    'coverage_percentage': float(np.count_nonzero(align_image) / align_image.size * 100)
                }
                align_analysis.append(align_info)
            
            analysis['workflow_stages']['aligned'] = {
                'total_files': len(aligned_files),
                'analyzed_files': len(align_analysis),
                'representative_analysis': align_analysis,
                'average_coverage': np.mean([a['coverage_percentage'] for a in align_analysis])
            }
            
            print(f"   🎯 Aligned: {len(aligned_files)} files, avg coverage: {analysis['workflow_stages']['aligned']['average_coverage']:.1f}%")
        
        # Compare processing stages
        if 'segmented' in analysis['workflow_stages'] and 'aligned' in analysis['workflow_stages']:
            seg_coverage = analysis['workflow_stages']['segmented']['average_coverage']
            align_coverage = analysis['workflow_stages']['aligned']['average_coverage']
            coverage_change = align_coverage - seg_coverage
            
            analysis['processing_quality']['alignment_effect'] = {
                'coverage_change_percentage': coverage_change,
                'alignment_impact': 'positive' if coverage_change > 0 else 'negative' if coverage_change < 0 else 'neutral'
            }
            
            print(f"   📈 Alignment Effect: {coverage_change:+.1f}% coverage change")
        
        return analysis
    
    def create_workflow_visualization(self, sample_info, analysis, output_dir):
        """Create comprehensive workflow visualization"""
        
        print(f"   📊 Creating workflow visualization...")
        
        # Create a comprehensive visualization showing all stages
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)
        
        # Row 1: Processing stages
        stage_titles = ['Raw Event', 'Segmented (Sample)', 'Aligned (Sample)', 'Processing Stats']
        
        # Raw event image
        if sample_info['raw_event_files']:
            ax1 = fig.add_subplot(gs[0, 0])
            raw_image = imread(sample_info['raw_event_files'][0])
            if raw_image.max() > raw_image.min():
                raw_norm = (raw_image - raw_image.min()) / (raw_image.max() - raw_image.min())
            else:
                raw_norm = raw_image
            im1 = ax1.imshow(raw_norm, cmap='hot', interpolation='nearest')
            ax1.set_title(f'{stage_titles[0]}\\n{raw_image.shape}')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Sample segmented image
        if sample_info['segmented_files']:
            ax2 = fig.add_subplot(gs[0, 1])
            seg_image = imread(sample_info['segmented_files'][len(sample_info['segmented_files'])//2])
            if seg_image.max() > seg_image.min():
                seg_norm = (seg_image - seg_image.min()) / (seg_image.max() - seg_image.min())
            else:
                seg_norm = seg_image
            im2 = ax2.imshow(seg_norm, cmap='hot', interpolation='nearest')
            ax2.set_title(f'{stage_titles[1]}\\n{len(sample_info["segmented_files"])} files')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Sample aligned image
        if sample_info['aligned_files']:
            ax3 = fig.add_subplot(gs[0, 2])
            align_image = imread(sample_info['aligned_files'][len(sample_info['aligned_files'])//2])
            if align_image.max() > align_image.min():
                align_norm = (align_image - align_image.min()) / (align_image.max() - align_image.min())
            else:
                align_norm = align_image
            im3 = ax3.imshow(align_norm, cmap='hot', interpolation='nearest')
            ax3.set_title(f'{stage_titles[2]}\\n{len(sample_info["aligned_files"])} files')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # Processing statistics
        ax4 = fig.add_subplot(gs[0, 3])
        if 'segmented' in analysis['workflow_stages'] and 'aligned' in analysis['workflow_stages']:
            seg_cov = analysis['workflow_stages']['segmented']['average_coverage']
            align_cov = analysis['workflow_stages']['aligned']['average_coverage']
            
            ax4.bar(['Segmented', 'Aligned'], [seg_cov, align_cov], 
                   color=['orange', 'green'], alpha=0.7)
            ax4.set_title('Average Coverage %')
            ax4.set_ylabel('Coverage (%)')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Processing\\nStatistics\\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Processing Stats')
        ax4.axis('off' if 'segmented' not in analysis['workflow_stages'] else 'on')
        
        # Row 2: File count analysis
        ax5 = fig.add_subplot(gs[1, :2])
        stages = ['Raw Event', 'Segmented', 'Aligned']
        counts = [
            len(sample_info['raw_event_files']),
            len(sample_info['segmented_files']),
            len(sample_info['aligned_files'])
        ]
        
        bars = ax5.bar(stages, counts, color=['red', 'orange', 'green'], alpha=0.7)
        ax5.set_title('Processing Stage File Counts')
        ax5.set_ylabel('Number of Files')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1, str(count),
                        ha='center', va='bottom', fontweight='bold')
        
        # Processing workflow diagram
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.text(0.5, 0.8, f'iQID PROCESSING WORKFLOW: {sample_info["sample_id"]}', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=14, fontweight='bold')
        
        workflow_text = "Raw Event Image → Segmentation (ROI Extraction) → Alignment → Final Stack"
        ax6.text(0.5, 0.6, workflow_text, ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12, color='blue')
        
        # Stage completion status
        stages = analysis.get('workflow_stages', {})
        status_text = f"Stages Available:\\n"
        status_text += f"✓ Raw Event: {len(sample_info['raw_event_files'])} files\\n"
        status_text += f"✓ Segmented: {len(sample_info['segmented_files'])} files\\n"
        status_text += f"✓ Aligned: {len(sample_info['aligned_files'])} files"
        
        ax6.text(0.5, 0.3, status_text, ha='center', va='center', 
                transform=ax6.transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        # Row 3: Detailed analysis
        if 'segmented' in analysis['workflow_stages']:
            ax7 = fig.add_subplot(gs[2, :2])
            seg_data = analysis['workflow_stages']['segmented']['representative_analysis']
            file_names = [s['file'][:10] + '...' for s in seg_data]  # Truncate names
            coverages = [s['coverage_percentage'] for s in seg_data]
            
            ax7.bar(range(len(file_names)), coverages, alpha=0.7, color='orange')
            ax7.set_title('Segmented File Coverage Analysis')
            ax7.set_xlabel('File Index')
            ax7.set_ylabel('Coverage (%)')
            ax7.set_xticks(range(len(file_names)))
            ax7.set_xticklabels([f'{i}' for i in range(len(file_names))], rotation=45)
            ax7.grid(True, alpha=0.3)
        
        if 'aligned' in analysis['workflow_stages']:
            ax8 = fig.add_subplot(gs[2, 2:])
            align_data = analysis['workflow_stages']['aligned']['representative_analysis']
            file_names = [a['file'][:10] + '...' for a in align_data]
            coverages = [a['coverage_percentage'] for a in align_data]
            
            ax8.bar(range(len(file_names)), coverages, alpha=0.7, color='green')
            ax8.set_title('Aligned File Coverage Analysis')
            ax8.set_xlabel('File Index')
            ax8.set_ylabel('Coverage (%)')
            ax8.set_xticks(range(len(file_names)))
            ax8.set_xticklabels([f'{i}' for i in range(len(file_names))], rotation=45)
            ax8.grid(True, alpha=0.3)
        
        plt.suptitle(f'iQID Processing Workflow Analysis: {sample_info["sample_id"]}', 
                     fontsize=16, fontweight='bold')
        
        # Save visualization
        viz_path = os.path.join(output_dir, f"{sample_info['sample_id']}_workflow_analysis.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"     ✓ Saved workflow visualization: {viz_path}")
        return viz_path
    
    def process_sample(self, sample_info):
        """Process a single iQID sample through workflow analysis"""
        
        print(f"\\n🚀 Processing sample: {sample_info['sample_id']}")
        
        # Create sample output directory
        sample_output = os.path.join(self.output_root, sample_info['sample_id'])
        os.makedirs(sample_output, exist_ok=True)
        
        # Analyze the complete workflow
        analysis = self.analyze_processing_workflow(sample_info)
        
        # Create visualization
        viz_path = self.create_workflow_visualization(sample_info, analysis, sample_output)
        
        # Save analysis report
        report = {
            'sample_info': sample_info,
            'workflow_analysis': analysis,
            'processing_timestamp': datetime.now().isoformat(),
            'pipeline_version': 'SimpleiQID_v1.0',
            'visualization_path': viz_path
        }
        
        report_path = os.path.join(sample_output, 'workflow_analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   ✓ Saved analysis report: {report_path}")
        
        return {
            'sample_id': sample_info['sample_id'],
            'output_dir': sample_output,
            'analysis': analysis,
            'report_path': report_path,
            'visualization_path': viz_path
        }
    
    def process_all_samples(self, max_samples=5):
        """Process all discovered samples"""
        
        print(f"\\n🚀 Starting batch iQID workflow analysis...")
        
        # Discover samples
        samples = self.discover_samples(max_samples)
        
        if not samples:
            print("   ❌ No samples found for processing")
            return []
        
        # Process each sample
        results = []
        for sample in samples:
            try:
                result = self.process_sample(sample)
                results.append(result)
            except Exception as e:
                print(f"   ❌ Error processing {sample['sample_id']}: {e}")
                continue
        
        # Create summary report
        self.create_batch_summary(results)
        
        print(f"\\n✅ Completed processing {len(results)} samples")
        return results
    
    def create_batch_summary(self, results):
        """Create batch processing summary"""
        
        print(f"\\n📋 Creating batch processing summary...")
        
        summary = {
            'batch_timestamp': datetime.now().isoformat(),
            'pipeline_version': 'SimpleiQID_v1.0',
            'total_samples': len(results),
            'successful_samples': len([r for r in results if r is not None]),
            'sample_summaries': []
        }
        
        for result in results:
            if result:
                analysis = result['analysis']
                sample_summary = {
                    'sample_id': result['sample_id'],
                    'workflow_stages': list(analysis['workflow_stages'].keys()),
                    'file_counts': {
                        'raw': len(analysis.get('workflow_stages', {}).get('raw_event', {}).get('file', [])) if 'raw_event' in analysis.get('workflow_stages', {}) else 0,
                        'segmented': analysis.get('workflow_stages', {}).get('segmented', {}).get('total_files', 0),
                        'aligned': analysis.get('workflow_stages', {}).get('aligned', {}).get('total_files', 0)
                    }
                }
                summary['sample_summaries'].append(sample_summary)
        
        # Save summary
        summary_path = os.path.join(self.output_root, 'batch_processing_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✓ Saved batch summary: {summary_path}")
        
        # Print summary
        print(f"\\n📊 BATCH PROCESSING SUMMARY:")
        print(f"   🔬 Total Samples: {summary['total_samples']}")
        print(f"   ✅ Successful: {summary['successful_samples']}")
        print(f"   📁 Output Directory: {self.output_root}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified iQID Processing Pipeline")
    parser.add_argument("--config", default="configs/iqid_pipeline_config.json",
                       help="Configuration file")
    parser.add_argument("--max-samples", type=int, default=5,
                       help="Maximum number of samples to process")
    parser.add_argument("--sample-id", type=str, help="Process specific sample by ID")
    
    args = parser.parse_args()
    
    pipeline = SimpleiQIDPipeline(args.config)
    
    if args.sample_id:
        # Process specific sample
        samples = pipeline.discover_samples(max_samples=20)
        target_sample = next((s for s in samples if s['sample_id'] == args.sample_id), None)
        if target_sample:
            result = pipeline.process_sample(target_sample)
            print(f"\\nProcessed sample: {result['sample_id']}")
        else:
            print(f"   ❌ Sample not found: {args.sample_id}")
            print(f"   Available samples: {[s['sample_id'] for s in samples]}")
    else:
        # Process all samples
        results = pipeline.process_all_samples(args.max_samples)
        print(f"\\nProcessed {len(results)} samples total")
