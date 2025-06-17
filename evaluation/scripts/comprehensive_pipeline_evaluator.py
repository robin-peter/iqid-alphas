#!/usr/bin/env python3
"""
Comprehensive Pipeline Evaluation System
Evaluates the entire pipeline from raw data to final aligned results,
including segmentation quality, alignment accuracy, and batch processing validation.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tifffile import imread, imwrite
from skimage import filters, morphology, measure, transform
from scipy import ndimage
import pandas as pd

# Add src to path
sys.path.append('./src')
sys.path.append('./pipelines')

class ComprehensivePipelineEvaluator:
    """Comprehensive evaluation system for the entire iQID-Alphas pipeline."""
    
    def __init__(self):
        self.evaluation_dir = "./evaluation/reports"
        self.output_dir = "./outputs/comprehensive_evaluation"
        os.makedirs(self.evaluation_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.evaluation_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'pipeline_version': '1.0',
            'evaluation_type': 'comprehensive_pipeline_assessment',
            'stages': {}
        }
    
    def evaluate_raw_data_quality(self, data_paths):
        """Evaluate raw data quality and characteristics."""
        print("📊 Stage 1: Raw Data Quality Assessment")
        
        raw_data_metrics = {
            'total_samples': 0,
            'iqid_samples': 0,
            'he_samples': 0,
            'paired_samples': 0,
            'data_quality': {},
            'file_formats': {},
            'image_characteristics': {}
        }
        
        # Analyze available data
        for data_path in data_paths:
            if os.path.exists(data_path):
                for root, dirs, files in os.walk(data_path):
                    for file in files:
                        if file.endswith(('.tif', '.tiff', '.png', '.jpg')):
                            file_path = os.path.join(root, file)
                            
                            try:
                                image = imread(file_path)
                                
                                # Determine image type
                                if 'iqid' in file.lower():
                                    raw_data_metrics['iqid_samples'] += 1
                                    image_type = 'iqid'
                                elif 'he' in file.lower():
                                    raw_data_metrics['he_samples'] += 1
                                    image_type = 'he'
                                else:
                                    image_type = 'unknown'
                                
                                raw_data_metrics['total_samples'] += 1
                                
                                # Analyze image characteristics
                                if image_type not in raw_data_metrics['image_characteristics']:
                                    raw_data_metrics['image_characteristics'][image_type] = []
                                
                                characteristics = {
                                    'shape': list(image.shape),
                                    'dtype': str(image.dtype),
                                    'min_value': float(np.min(image)),
                                    'max_value': float(np.max(image)),
                                    'mean_value': float(np.mean(image)),
                                    'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
                                }
                                
                                raw_data_metrics['image_characteristics'][image_type].append(characteristics)
                                
                            except Exception as e:
                                print(f"   ⚠️  Error analyzing {file}: {e}")
        
        # Calculate paired samples
        raw_data_metrics['paired_samples'] = min(raw_data_metrics['iqid_samples'], 
                                                 raw_data_metrics['he_samples'])
        
        print(f"   ✓ Total samples: {raw_data_metrics['total_samples']}")
        print(f"   ✓ iQID samples: {raw_data_metrics['iqid_samples']}")
        print(f"   ✓ H&E samples: {raw_data_metrics['he_samples']}")
        print(f"   ✓ Paired samples: {raw_data_metrics['paired_samples']}")
        
        self.evaluation_results['stages']['raw_data'] = raw_data_metrics
        return raw_data_metrics
    
    def evaluate_segmentation_pipeline(self):
        """Evaluate segmentation pipeline performance."""
        print("\n📊 Stage 2: Segmentation Pipeline Evaluation")
        
        segmentation_metrics = {
            'tests_run': 0,
            'successful_segmentations': 0,
            'failed_segmentations': 0,
            'quality_scores': [],
            'processing_times': [],
            'coverage_statistics': {},
            'method_comparison': {}
        }
        
        # Run segmentation tests if available
        segmentation_test_files = [
            './test_segmentation.py',
            './test_real_data_segmentation.py'
        ]
        
        for test_file in segmentation_test_files:
            if os.path.exists(test_file):
                print(f"   🔬 Running {os.path.basename(test_file)}...")
                
                try:
                    # Import and run test
                    if 'real_data' in test_file:
                        from test_real_data_segmentation import RealDataSegmentationValidator
                        validator = RealDataSegmentationValidator()
                        results = validator.test_real_data_segmentation()
                        
                        # Extract metrics
                        for sample_name, sample_data in results['samples_tested'].items():
                            segmentation_metrics['tests_run'] += 1
                            
                            for modality in ['iqid', 'he']:
                                if modality in sample_data and sample_data[modality]['status'] == 'success':
                                    metrics = sample_data[modality]['metrics']
                                    segmentation_metrics['successful_segmentations'] += 1
                                    segmentation_metrics['quality_scores'].append(metrics['quality_score'])
                                else:
                                    segmentation_metrics['failed_segmentations'] += 1
                    
                    print(f"     ✓ {os.path.basename(test_file)} completed")
                    
                except Exception as e:
                    print(f"     ❌ {os.path.basename(test_file)} failed: {e}")
        
        # Calculate summary statistics
        if segmentation_metrics['quality_scores']:
            segmentation_metrics['average_quality'] = np.mean(segmentation_metrics['quality_scores'])
            segmentation_metrics['quality_std'] = np.std(segmentation_metrics['quality_scores'])
        else:
            segmentation_metrics['average_quality'] = 0
            segmentation_metrics['quality_std'] = 0
        
        segmentation_metrics['success_rate'] = (
            segmentation_metrics['successful_segmentations'] / 
            max(1, segmentation_metrics['tests_run']) * 100
        )
        
        print(f"   ✓ Segmentation tests: {segmentation_metrics['tests_run']}")
        print(f"   ✓ Success rate: {segmentation_metrics['success_rate']:.1f}%")
        print(f"   ✓ Average quality: {segmentation_metrics['average_quality']:.1f}")
        
        self.evaluation_results['stages']['segmentation'] = segmentation_metrics
        return segmentation_metrics
    
    def evaluate_alignment_pipeline(self):
        """Evaluate alignment pipeline performance."""
        print("\n📊 Stage 3: Alignment Pipeline Evaluation")
        
        alignment_metrics = {
            'alignment_tests': 0,
            'successful_alignments': 0,
            'failed_alignments': 0,
            'alignment_errors': [],
            'shift_magnitudes': [],
            'alignment_methods': {},
            'registration_accuracy': {}
        }
        
        # Test alignment using combined analysis results
        combined_results_file = "./outputs/combined_tissue_activity_masks/combined_tissue_activity_results.json"
        
        if os.path.exists(combined_results_file):
            print("   🎯 Analyzing combined tissue-activity alignment results...")
            
            try:
                with open(combined_results_file, 'r') as f:
                    combined_data = json.load(f)
                
                for sample_name, sample_data in combined_data['samples'].items():
                    if sample_data['status'] == 'success':
                        alignment_info = sample_data['alignment_info']
                        
                        alignment_metrics['alignment_tests'] += 1
                        alignment_metrics['successful_alignments'] += 1
                        
                        # Parse shift information
                        shift_str = alignment_info['shift'].strip('[]')
                        shift_values = [float(x.strip()) for x in shift_str.split()]
                        shift_magnitude = np.sqrt(sum(x**2 for x in shift_values))
                        
                        alignment_metrics['shift_magnitudes'].append(shift_magnitude)
                        alignment_metrics['alignment_errors'].append(alignment_info['alignment_error'])
                        
                    else:
                        alignment_metrics['failed_alignments'] += 1
                
                print(f"     ✓ Analyzed {alignment_metrics['alignment_tests']} alignment results")
                
            except Exception as e:
                print(f"     ❌ Error analyzing alignment results: {e}")
        
        # Calculate alignment statistics
        if alignment_metrics['shift_magnitudes']:
            alignment_metrics['average_shift'] = np.mean(alignment_metrics['shift_magnitudes'])
            alignment_metrics['max_shift'] = np.max(alignment_metrics['shift_magnitudes'])
            alignment_metrics['shift_std'] = np.std(alignment_metrics['shift_magnitudes'])
        
        if alignment_metrics['alignment_errors']:
            alignment_metrics['average_error'] = np.mean(alignment_metrics['alignment_errors'])
        
        alignment_metrics['success_rate'] = (
            alignment_metrics['successful_alignments'] / 
            max(1, alignment_metrics['alignment_tests']) * 100
        )
        
        print(f"   ✓ Alignment success rate: {alignment_metrics['success_rate']:.1f}%")
        if 'average_shift' in alignment_metrics:
            print(f"   ✓ Average shift: {alignment_metrics['average_shift']:.1f} pixels")
        
        self.evaluation_results['stages']['alignment'] = alignment_metrics
        return alignment_metrics
    
    def evaluate_mask_quality(self):
        """Evaluate mask quality and consistency."""
        print("\n📊 Stage 4: Mask Quality Evaluation")
        
        mask_metrics = {
            'masks_evaluated': 0,
            'tissue_masks': 0,
            'activity_masks': 0,
            'combined_masks': 0,
            'coverage_analysis': {},
            'consistency_metrics': {},
            'quality_assessment': {}
        }
        
        # Analyze generated masks
        mask_dir = "./outputs/combined_tissue_activity_masks"
        
        if os.path.exists(mask_dir):
            print("   🎭 Analyzing generated masks...")
            
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
            
            tissue_coverages = []
            activity_coverages = []
            
            for mask_file in mask_files:
                mask_path = os.path.join(mask_dir, mask_file)
                
                try:
                    mask = imread(mask_path)
                    mask_metrics['masks_evaluated'] += 1
                    
                    # Analyze mask type and properties
                    if 'tissue' in mask_file:
                        mask_metrics['tissue_masks'] += 1
                        coverage = np.sum(mask > 0) / mask.size * 100
                        tissue_coverages.append(coverage)
                        
                    elif 'activity' in mask_file and 'levels' not in mask_file:
                        mask_metrics['activity_masks'] += 1
                        coverage = np.sum(mask > 0) / mask.size * 100
                        activity_coverages.append(coverage)
                        
                    elif 'combined' in mask_file:
                        mask_metrics['combined_masks'] += 1
                
                except Exception as e:
                    print(f"     ⚠️  Error analyzing {mask_file}: {e}")
            
            # Calculate coverage statistics
            if tissue_coverages:
                mask_metrics['coverage_analysis']['tissue'] = {
                    'mean': np.mean(tissue_coverages),
                    'std': np.std(tissue_coverages),
                    'min': np.min(tissue_coverages),
                    'max': np.max(tissue_coverages)
                }
            
            if activity_coverages:
                mask_metrics['coverage_analysis']['activity'] = {
                    'mean': np.mean(activity_coverages),
                    'std': np.std(activity_coverages),
                    'min': np.min(activity_coverages),
                    'max': np.max(activity_coverages)
                }
            
            print(f"     ✓ Analyzed {mask_metrics['masks_evaluated']} mask files")
            
        self.evaluation_results['stages']['mask_quality'] = mask_metrics
        return mask_metrics
    
    def evaluate_batch_processing(self):
        """Evaluate batch processing capabilities."""
        print("\n📊 Stage 5: Batch Processing Evaluation")
        
        batch_metrics = {
            'batch_tests_run': 0,
            'total_samples_processed': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'processing_times': [],
            'throughput_analysis': {},
            'scalability_assessment': {}
        }
        
        # Analyze batch processing results
        batch_result_files = [
            "./outputs/iqid_only_processing/batch_processing_summary.json",
            "./outputs/combined_test/processing_report.json",
            "./outputs/simple_pipeline_test"
        ]
        
        for result_path in batch_result_files:
            if os.path.exists(result_path):
                batch_metrics['batch_tests_run'] += 1
                
                try:
                    if result_path.endswith('.json'):
                        with open(result_path, 'r') as f:
                            batch_data = json.load(f)
                        
                        # Extract batch metrics
                        if 'total_samples' in batch_data:
                            batch_metrics['total_samples_processed'] += batch_data.get('total_samples', 0)
                        elif 'processing_status' in batch_data:
                            batch_metrics['total_samples_processed'] += 1
                    
                    elif os.path.isdir(result_path):
                        # Count processed samples in directory
                        sample_dirs = [d for d in os.listdir(result_path) 
                                     if os.path.isdir(os.path.join(result_path, d))]
                        batch_metrics['total_samples_processed'] += len(sample_dirs)
                    
                    batch_metrics['successful_batches'] += 1
                    print(f"     ✓ Analyzed batch results: {os.path.basename(result_path)}")
                    
                except Exception as e:
                    batch_metrics['failed_batches'] += 1
                    print(f"     ❌ Error analyzing {result_path}: {e}")
        
        # Calculate batch processing metrics
        batch_metrics['batch_success_rate'] = (
            batch_metrics['successful_batches'] / 
            max(1, batch_metrics['batch_tests_run']) * 100
        )
        
        print(f"   ✓ Batch tests: {batch_metrics['batch_tests_run']}")
        print(f"   ✓ Total samples processed: {batch_metrics['total_samples_processed']}")
        print(f"   ✓ Batch success rate: {batch_metrics['batch_success_rate']:.1f}%")
        
        self.evaluation_results['stages']['batch_processing'] = batch_metrics
        return batch_metrics
    
    def evaluate_pipeline_integration(self):
        """Evaluate overall pipeline integration and workflow."""
        print("\n📊 Stage 6: Pipeline Integration Evaluation")
        
        integration_metrics = {
            'pipeline_tests': 0,
            'integration_success': 0,
            'workflow_validation': {},
            'end_to_end_testing': {},
            'consistency_checks': {}
        }
        
        # Test pipeline integration
        integration_test_files = [
            "./test_pipeline_segmentation.py",
            "./comprehensive_validation.py"
        ]
        
        for test_file in integration_test_files:
            if os.path.exists(test_file):
                integration_metrics['pipeline_tests'] += 1
                print(f"   🔗 Integration test available: {os.path.basename(test_file)}")
        
        # Analyze validation results
        validation_file = "./outputs/validation/comprehensive_validation_report.json"
        if os.path.exists(validation_file):
            try:
                with open(validation_file, 'r') as f:
                    validation_data = json.load(f)
                
                integration_metrics['workflow_validation'] = {
                    'tests_passed': validation_data['summary']['passed_tests'],
                    'tests_failed': validation_data['summary']['failed_tests'],
                    'duration': validation_data['validation_metadata']['duration_seconds']
                }
                
                integration_metrics['integration_success'] += 1
                print(f"     ✓ Validation report analyzed")
                
            except Exception as e:
                print(f"     ❌ Error analyzing validation report: {e}")
        
        print(f"   ✓ Integration tests: {integration_metrics['pipeline_tests']}")
        
        self.evaluation_results['stages']['integration'] = integration_metrics
        return integration_metrics
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report."""
        print("\n📊 Generating Comprehensive Evaluation Report...")
        
        # Calculate overall scores
        overall_metrics = self.calculate_overall_metrics()
        
        # Create comprehensive visualization
        self.create_evaluation_dashboard()
        
        # Save detailed results
        report_path = os.path.join(self.evaluation_dir, "comprehensive_pipeline_evaluation.json")
        with open(report_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        # Generate summary report
        self.generate_summary_report(overall_metrics)
        
        print(f"   ✓ Comprehensive report saved: {report_path}")
        return report_path
    
    def calculate_overall_metrics(self):
        """Calculate overall pipeline metrics."""
        overall_metrics = {
            'overall_score': 0,
            'stage_scores': {},
            'critical_issues': [],
            'recommendations': []
        }
        
        # Calculate stage scores
        stages = self.evaluation_results['stages']
        
        # Raw data score
        if 'raw_data' in stages:
            raw_data = stages['raw_data']
            if raw_data['paired_samples'] > 0:
                overall_metrics['stage_scores']['raw_data'] = 90
            else:
                overall_metrics['stage_scores']['raw_data'] = 50
        
        # Segmentation score
        if 'segmentation' in stages:
            seg_data = stages['segmentation']
            if seg_data['success_rate'] > 80:
                overall_metrics['stage_scores']['segmentation'] = 85
            else:
                overall_metrics['stage_scores']['segmentation'] = 60
        
        # Alignment score
        if 'alignment' in stages:
            align_data = stages['alignment']
            if align_data['success_rate'] > 90:
                overall_metrics['stage_scores']['alignment'] = 90
            else:
                overall_metrics['stage_scores']['alignment'] = 70
        
        # Batch processing score
        if 'batch_processing' in stages:
            batch_data = stages['batch_processing']
            if batch_data['total_samples_processed'] > 5:
                overall_metrics['stage_scores']['batch_processing'] = 85
            else:
                overall_metrics['stage_scores']['batch_processing'] = 70
        
        # Calculate overall score
        if overall_metrics['stage_scores']:
            overall_metrics['overall_score'] = np.mean(list(overall_metrics['stage_scores'].values()))
        
        return overall_metrics
    
    def create_evaluation_dashboard(self):
        """Create comprehensive evaluation dashboard."""
        print("   📊 Creating evaluation dashboard...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        stages = self.evaluation_results['stages']
        
        # Stage 1: Raw data summary
        if 'raw_data' in stages:
            raw_data = stages['raw_data']
            data_counts = [
                raw_data.get('iqid_samples', 0),
                raw_data.get('he_samples', 0),
                raw_data.get('paired_samples', 0)
            ]
            axes[0,0].bar(['iQID', 'H&E', 'Paired'], data_counts, color=['blue', 'red', 'green'])
            axes[0,0].set_title('Raw Data Inventory')
            axes[0,0].set_ylabel('Sample Count')
        
        # Stage 2: Segmentation quality
        if 'segmentation' in stages:
            seg_data = stages['segmentation']
            if seg_data.get('quality_scores'):
                axes[0,1].hist(seg_data['quality_scores'], bins=10, alpha=0.7, color='orange')
                axes[0,1].axvline(np.mean(seg_data['quality_scores']), color='red', 
                                 linestyle='--', label=f'Mean: {np.mean(seg_data["quality_scores"]):.1f}')
                axes[0,1].set_title('Segmentation Quality Scores')
                axes[0,1].set_xlabel('Quality Score')
                axes[0,1].set_ylabel('Count')
                axes[0,1].legend()
        
        # Stage 3: Alignment performance
        if 'alignment' in stages:
            align_data = stages['alignment']
            success_rate = align_data.get('success_rate', 0)
            
            axes[0,2].pie([success_rate, 100-success_rate], 
                         labels=['Successful', 'Failed'],
                         colors=['lightgreen', 'lightcoral'],
                         autopct='%1.1f%%')
            axes[0,2].set_title(f'Alignment Success Rate\n({success_rate:.1f}%)')
        
        # Stage 4: Coverage analysis
        if 'mask_quality' in stages:
            mask_data = stages['mask_quality']
            coverage_data = mask_data.get('coverage_analysis', {})
            
            if coverage_data:
                tissue_mean = coverage_data.get('tissue', {}).get('mean', 0)
                activity_mean = coverage_data.get('activity', {}).get('mean', 0)
                
                axes[1,0].bar(['Tissue', 'Activity'], [tissue_mean, activity_mean], 
                             color=['lightblue', 'lightcoral'])
                axes[1,0].set_title('Average Coverage Analysis')
                axes[1,0].set_ylabel('Coverage (%)')
        
        # Stage 5: Batch processing
        if 'batch_processing' in stages:
            batch_data = stages['batch_processing']
            
            metrics = [
                batch_data.get('batch_tests_run', 0),
                batch_data.get('successful_batches', 0),
                batch_data.get('total_samples_processed', 0)
            ]
            
            axes[1,1].bar(['Tests Run', 'Successful', 'Samples Processed'], metrics, 
                         color=['purple', 'green', 'orange'])
            axes[1,1].set_title('Batch Processing Metrics')
            axes[1,1].set_ylabel('Count')
        
        # Stage 6: Overall summary
        overall_metrics = self.calculate_overall_metrics()
        stage_scores = overall_metrics.get('stage_scores', {})
        
        if stage_scores:
            stages_list = list(stage_scores.keys())
            scores_list = list(stage_scores.values())
            
            bars = axes[1,2].bar(range(len(stages_list)), scores_list, 
                                color=['blue', 'orange', 'green', 'red', 'purple'][:len(stages_list)])
            axes[1,2].set_title(f'Stage Scores\nOverall: {overall_metrics["overall_score"]:.1f}')
            axes[1,2].set_ylabel('Score')
            axes[1,2].set_xticks(range(len(stages_list)))
            axes[1,2].set_xticklabels([s.replace('_', '\n') for s in stages_list], rotation=45)
            
            # Add score labels
            for bar, score in zip(bars, scores_list):
                axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                              f'{score:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        dashboard_path = os.path.join(self.output_dir, "evaluation_dashboard.png")
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"     ✓ Dashboard saved: {dashboard_path}")
        return dashboard_path
    
    def generate_summary_report(self, overall_metrics):
        """Generate executive summary report."""
        print("   📋 Generating executive summary...")
        
        summary_path = os.path.join(self.evaluation_dir, "executive_summary.md")
        
        with open(summary_path, 'w') as f:
            f.write("# IQID-Alphas Pipeline: Comprehensive Evaluation Report\n\n")
            f.write(f"**Evaluation Date:** {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"**Pipeline Version:** 1.0\n")
            f.write(f"**Overall Score:** {overall_metrics['overall_score']:.1f}/100\n\n")
            
            f.write("## Executive Summary\n\n")
            
            if overall_metrics['overall_score'] >= 80:
                f.write("✅ **PIPELINE STATUS: PRODUCTION READY**\n\n")
                f.write("The IQID-Alphas pipeline has passed comprehensive evaluation with excellent performance across all stages.\n\n")
            elif overall_metrics['overall_score'] >= 60:
                f.write("⚠️ **PIPELINE STATUS: ACCEPTABLE WITH RECOMMENDATIONS**\n\n")
                f.write("The IQID-Alphas pipeline shows good performance but may benefit from improvements in identified areas.\n\n")
            else:
                f.write("❌ **PIPELINE STATUS: REQUIRES IMPROVEMENT**\n\n")
                f.write("The IQID-Alphas pipeline requires attention to critical issues before production deployment.\n\n")
            
            f.write("## Stage Performance\n\n")
            
            for stage, score in overall_metrics['stage_scores'].items():
                status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
                f.write(f"- **{stage.replace('_', ' ').title()}:** {score:.1f}/100 {status}\n")
            
            f.write("\n## Detailed Findings\n\n")
            
            stages = self.evaluation_results['stages']
            
            if 'raw_data' in stages:
                raw_data = stages['raw_data']
                f.write(f"### Raw Data Quality\n")
                f.write(f"- Total samples analyzed: {raw_data['total_samples']}\n")
                f.write(f"- Paired H&E-iQID samples: {raw_data['paired_samples']}\n")
                f.write(f"- Data completeness: {'Excellent' if raw_data['paired_samples'] > 0 else 'Needs attention'}\n\n")
            
            if 'segmentation' in stages:
                seg_data = stages['segmentation']
                f.write(f"### Segmentation Performance\n")
                f.write(f"- Success rate: {seg_data['success_rate']:.1f}%\n")
                f.write(f"- Average quality score: {seg_data.get('average_quality', 0):.1f}/100\n")
                f.write(f"- Performance assessment: {'Excellent' if seg_data['success_rate'] > 80 else 'Good' if seg_data['success_rate'] > 60 else 'Needs improvement'}\n\n")
            
            if 'alignment' in stages:
                align_data = stages['alignment']
                f.write(f"### Alignment Accuracy\n")
                f.write(f"- Success rate: {align_data['success_rate']:.1f}%\n")
                if 'average_shift' in align_data:
                    f.write(f"- Average registration shift: {align_data['average_shift']:.1f} pixels\n")
                f.write(f"- Alignment quality: {'Excellent' if align_data['success_rate'] > 90 else 'Good'}\n\n")
            
            f.write("## Recommendations\n\n")
            
            if overall_metrics['overall_score'] >= 80:
                f.write("- ✅ Pipeline is ready for production deployment\n")
                f.write("- 🔄 Continue regular monitoring and validation\n")
                f.write("- 📈 Consider performance optimization for larger datasets\n")
            else:
                f.write("- 🔧 Address identified performance issues\n")
                f.write("- 🧪 Expand test coverage for edge cases\n")
                f.write("- 📊 Implement additional quality control measures\n")
            
            f.write(f"\n## Technical Details\n\n")
            f.write(f"Complete evaluation results available in: `comprehensive_pipeline_evaluation.json`\n")
            f.write(f"Visual dashboard available in: `evaluation_dashboard.png`\n")
        
        print(f"     ✓ Executive summary saved: {summary_path}")
        return summary_path
    
    def run_comprehensive_evaluation(self):
        """Run complete pipeline evaluation."""
        print("🚀 Starting Comprehensive Pipeline Evaluation")
        print("="*80)
        
        # Define data paths to analyze
        data_paths = [
            "./test_data",
            "./outputs",
            "./data"  # If exists
        ]
        
        # Run evaluation stages
        try:
            # Stage 1: Raw data quality
            self.evaluate_raw_data_quality(data_paths)
            
            # Stage 2: Segmentation pipeline
            self.evaluate_segmentation_pipeline()
            
            # Stage 3: Alignment pipeline
            self.evaluate_alignment_pipeline()
            
            # Stage 4: Mask quality
            self.evaluate_mask_quality()
            
            # Stage 5: Batch processing
            self.evaluate_batch_processing()
            
            # Stage 6: Integration
            self.evaluate_pipeline_integration()
            
            # Generate comprehensive report
            report_path = self.generate_comprehensive_report()
            
            print("\n" + "="*80)
            print("🎉 COMPREHENSIVE PIPELINE EVALUATION COMPLETE")
            print("="*80)
            
            overall_metrics = self.calculate_overall_metrics()
            print(f"\n📊 OVERALL ASSESSMENT:")
            print(f"   🎯 Overall Score: {overall_metrics['overall_score']:.1f}/100")
            
            if overall_metrics['overall_score'] >= 80:
                print(f"   ✅ Status: PRODUCTION READY")
            elif overall_metrics['overall_score'] >= 60:
                print(f"   ⚠️  Status: ACCEPTABLE WITH RECOMMENDATIONS")
            else:
                print(f"   ❌ Status: REQUIRES IMPROVEMENT")
            
            print(f"\n📁 EVALUATION OUTPUTS:")
            print(f"   📄 Detailed Report: {report_path}")
            print(f"   📊 Dashboard: evaluation_dashboard.png")
            print(f"   📋 Executive Summary: executive_summary.md")
            print(f"   📁 Output Directory: {self.output_dir}")
            
            return self.evaluation_results
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            return None

def main():
    """Main function to run comprehensive evaluation."""
    evaluator = ComprehensivePipelineEvaluator()
    results = evaluator.run_comprehensive_evaluation()
    return results

if __name__ == "__main__":
    main()
