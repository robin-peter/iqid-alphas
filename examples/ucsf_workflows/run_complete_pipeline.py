#!/usr/bin/env python3
"""
UCSF Complete Processing Pipeline

Master workflow that orchestrates both iQID alignment and H&E co-registration
workflows for comprehensive UCSF dataset analysis.

This script:
1. Runs Workflow 1: iQID raw → aligned
2. Runs Workflow 2: aligned iQID + H&E → co-registered analysis
3. Generates comprehensive reports and visualizations

Author: Wookjin Choi <wookjin.choi@jefferson.edu>
Date: June 2025
"""

import sys
import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Import the workflow modules
from workflow1_iqid_alignment import UCSFiQIDWorkflow
from workflow2_he_iqid_coregistration import UCSFHEiQIDWorkflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('intermediate/complete_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UCSFCompletePipeline:
    """
    Master pipeline orchestrating both iQID alignment and H&E co-registration workflows.
    """
    
    def __init__(self):
        """Initialize the complete pipeline."""
        self.workflow1 = UCSFiQIDWorkflow("configs/unified_config.json")
        self.workflow2 = UCSFHEiQIDWorkflow("configs/unified_config.json")
        
        # Setup master output directory
        self.master_output_dir = Path("outputs/complete_analysis")
        self.master_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("UCSF Complete Pipeline initialized")
    
    def run_complete_analysis(self, raw_iqid_path: str, he_images_path: str) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            raw_iqid_path: Path to raw iQID data
            he_images_path: Path to H&E histology images
            
        Returns:
            Dictionary containing results from both workflows
        """
        logger.info("Starting UCSF Complete Analysis Pipeline")
        pipeline_start_time = time.time()
        
        try:
            # Stage 1: iQID Alignment (Workflow 1)
            logger.info("=" * 60)
            logger.info("STAGE 1: iQID Raw Data Alignment")
            logger.info("=" * 60)
            
            workflow1_start = time.time()
            workflow1_results = self.workflow1.run_complete_workflow(raw_iqid_path)
            workflow1_time = time.time() - workflow1_start
            
            logger.info(f"Stage 1 completed in {workflow1_time:.1f} seconds")
            
            # Stage 2: H&E-iQID Co-registration (Workflow 2)
            logger.info("=" * 60)
            logger.info("STAGE 2: H&E-iQID Co-registration")
            logger.info("=" * 60)
            
            workflow2_start = time.time()
            # Use the aligned iQID output from workflow 1
            aligned_iqid_path = "outputs/iqid_aligned/aligned_iqid_stack.npy"
            workflow2_results = self.workflow2.run_complete_workflow(
                aligned_iqid_path, he_images_path
            )
            workflow2_time = time.time() - workflow2_start
            
            logger.info(f"Stage 2 completed in {workflow2_time:.1f} seconds")
            
            # Stage 3: Generate comprehensive report
            logger.info("=" * 60)
            logger.info("STAGE 3: Comprehensive Report Generation")
            logger.info("=" * 60)
            
            comprehensive_results = self.generate_comprehensive_report(
                workflow1_results, workflow2_results, 
                workflow1_time, workflow2_time
            )
            
            total_time = time.time() - pipeline_start_time
            logger.info(f"Complete pipeline finished in {total_time:.1f} seconds")
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Complete pipeline failed: {e}")
            raise
    
    def generate_comprehensive_report(self, workflow1_results: Dict[str, Any],
                                    workflow2_results: Dict[str, Any],
                                    w1_time: float, w2_time: float) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report combining both workflows.
        
        Args:
            workflow1_results: Results from iQID alignment workflow
            workflow2_results: Results from H&E co-registration workflow
            w1_time: Workflow 1 execution time
            w2_time: Workflow 2 execution time
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info("Generating comprehensive analysis report")
        
        # Extract key metrics from both workflows
        w1_metrics = workflow1_results.get('alignment_metrics', {})
        w2_analysis = workflow2_results.get('analysis_data', {})
        w2_registration = workflow2_results.get('registration_data', {})
        
        # Create comprehensive summary
        comprehensive_summary = {
            "pipeline_metadata": {
                "pipeline_name": "UCSF Complete iQID-H&E Analysis",
                "version": "1.0.0",
                "execution_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_processing_time": w1_time + w2_time,
                "workflow1_time": w1_time,
                "workflow2_time": w2_time
            },
            
            "workflow1_summary": {
                "name": "iQID Alignment",
                "status": "completed",
                "n_frames_processed": len(workflow1_results.get('aligned_stack', [])),
                "reference_frame": w1_metrics.get('reference_frame', 'unknown'),
                "mean_correlation": float(np.mean(w1_metrics.get('correlations', [0]))) if w1_metrics.get('correlations') else 0,
                "alignment_quality": "good" if float(np.mean(w1_metrics.get('correlations', [0]))) > 0.7 else "fair"
            },
            
            "workflow2_summary": {
                "name": "H&E-iQID Co-registration",
                "status": "completed",
                "registration_correlation": w2_registration.get('registration_metrics', {}).get('correlation_after', 0),
                "n_tissue_regions": w2_analysis.get('overall_stats', {}).get('n_tissue_regions', 0),
                "total_tissue_area": w2_analysis.get('overall_stats', {}).get('total_tissue_area', 0),
                "activity_density": w2_analysis.get('overall_stats', {}).get('activity_density', 0)
            },
            
            "integrated_analysis": {
                "total_activity": w2_analysis.get('overall_stats', {}).get('total_activity', 0),
                "tissue_coverage": w2_analysis.get('overall_stats', {}).get('total_tissue_area', 0),
                "analysis_quality": self.assess_analysis_quality(workflow1_results, workflow2_results),
                "key_findings": self.extract_key_findings(workflow1_results, workflow2_results)
            },
            
            "file_outputs": {
                "aligned_iqid_stack": "outputs/iqid_aligned/aligned_iqid_stack.npy",
                "tissue_activity_data": "outputs/he_iqid_analysis/tissue_activity_data.csv",
                "quantitative_analysis": "outputs/he_iqid_analysis/quantitative_analysis.json",
                "visualizations_dir": "outputs/he_iqid_analysis/visualizations/",
                "processing_logs": [
                    "intermediate/workflow1_iqid_alignment.log",
                    "intermediate/workflow2_he_iqid_coregistration.log",
                    "intermediate/complete_pipeline.log"
                ]
            },
            
            "quality_assessment": {
                "workflow1_quality": self.assess_workflow1_quality(workflow1_results),
                "workflow2_quality": self.assess_workflow2_quality(workflow2_results),
                "overall_quality": "good",  # Would be calculated based on both workflows
                "recommendations": self.generate_recommendations(workflow1_results, workflow2_results)
            }
        }
        
        # Save comprehensive report
        report_path = self.master_output_dir / "comprehensive_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(comprehensive_summary, f, indent=2, default=str)
        
        logger.info(f"Comprehensive report saved to {report_path}")
        
        # Also create a human-readable summary
        self.create_human_readable_summary(comprehensive_summary)
        
        return {
            "workflow1_results": workflow1_results,
            "workflow2_results": workflow2_results,
            "comprehensive_summary": comprehensive_summary
        }
    
    def assess_analysis_quality(self, w1_results: Dict[str, Any], 
                               w2_results: Dict[str, Any]) -> str:
        """Assess overall analysis quality."""
        # Simplified quality assessment
        w1_corr = np.mean(w1_results.get('alignment_metrics', {}).get('correlations', [0]))
        w2_corr = w2_results.get('registration_data', {}).get('registration_metrics', {}).get('correlation_after', 0)
        
        if w1_corr > 0.8 and w2_corr > 0.6:
            return "excellent"
        elif w1_corr > 0.7 and w2_corr > 0.5:
            return "good"
        elif w1_corr > 0.6 and w2_corr > 0.4:
            return "fair"
        else:
            return "poor"
    
    def extract_key_findings(self, w1_results: Dict[str, Any], 
                           w2_results: Dict[str, Any]) -> List[str]:
        """Extract key findings from the analysis."""
        findings = []
        
        # From workflow 1
        w1_metrics = w1_results.get('alignment_metrics', {})
        if w1_metrics.get('correlations'):
            mean_corr = np.mean(w1_metrics['correlations'])
            findings.append(f"iQID frames aligned with mean correlation of {mean_corr:.3f}")
        
        # From workflow 2
        w2_analysis = w2_results.get('analysis_data', {})
        if w2_analysis.get('overall_stats'):
            stats = w2_analysis['overall_stats']
            findings.append(f"Identified {stats.get('n_tissue_regions', 0)} tissue regions")
            findings.append(f"Average activity density: {stats.get('activity_density', 0):.3f}")
        
        return findings
    
    def assess_workflow1_quality(self, w1_results: Dict[str, Any]) -> Dict[str, str]:
        """Assess Workflow 1 quality."""
        metrics = w1_results.get('alignment_metrics', {})
        correlations = metrics.get('correlations', [])
        
        if correlations:
            mean_corr = np.mean(correlations)
            if mean_corr > 0.8:
                quality = "excellent"
            elif mean_corr > 0.7:
                quality = "good"
            else:
                quality = "fair"
        else:
            quality = "unknown"
        
        return {
            "overall": quality,
            "alignment_correlation": f"{np.mean(correlations):.3f}" if correlations else "N/A",
            "n_frames": str(len(correlations)) if correlations else "N/A"
        }
    
    def assess_workflow2_quality(self, w2_results: Dict[str, Any]) -> Dict[str, str]:
        """Assess Workflow 2 quality."""
        registration = w2_results.get('registration_data', {})
        analysis = w2_results.get('analysis_data', {})
        
        reg_corr = registration.get('registration_metrics', {}).get('correlation_after', 0)
        n_tissues = analysis.get('overall_stats', {}).get('n_tissue_regions', 0)
        
        if reg_corr > 0.6 and n_tissues > 0:
            quality = "good"
        elif reg_corr > 0.4:
            quality = "fair"
        else:
            quality = "poor"
        
        return {
            "overall": quality,
            "registration_correlation": f"{reg_corr:.3f}",
            "tissue_regions_found": str(n_tissues)
        }
    
    def generate_recommendations(self, w1_results: Dict[str, Any], 
                               w2_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        # Check workflow 1 quality
        w1_corr = np.mean(w1_results.get('alignment_metrics', {}).get('correlations', [0]))
        if w1_corr < 0.7:
            recommendations.append("Consider adjusting iQID alignment parameters for better frame correlation")
        
        # Check workflow 2 quality
        w2_corr = w2_results.get('registration_data', {}).get('registration_metrics', {}).get('correlation_after', 0)
        if w2_corr < 0.5:
            recommendations.append("H&E-iQID registration may need manual inspection or parameter adjustment")
        
        # Check tissue detection
        n_tissues = w2_results.get('analysis_data', {}).get('overall_stats', {}).get('n_tissue_regions', 0)
        if n_tissues == 0:
            recommendations.append("No tissue regions detected - check segmentation parameters")
        
        return recommendations
    
    def create_human_readable_summary(self, summary: Dict[str, Any]) -> None:
        """Create a human-readable summary report."""
        summary_text = f"""
UCSF iQID-H&E Analysis Pipeline Summary
======================================

Analysis Date: {summary['pipeline_metadata']['execution_timestamp']}
Total Processing Time: {summary['pipeline_metadata']['total_processing_time']:.1f} seconds

WORKFLOW 1 RESULTS (iQID Alignment)
-----------------------------------
• Status: {summary['workflow1_summary']['status'].upper()}
• Frames Processed: {summary['workflow1_summary']['n_frames_processed']}
• Mean Alignment Correlation: {summary['workflow1_summary']['mean_correlation']:.3f}
• Quality Assessment: {summary['workflow1_summary']['alignment_quality'].upper()}

WORKFLOW 2 RESULTS (H&E-iQID Co-registration)
--------------------------------------------
• Status: {summary['workflow2_summary']['status'].upper()}
• Registration Correlation: {summary['workflow2_summary']['registration_correlation']:.3f}
• Tissue Regions Identified: {summary['workflow2_summary']['n_tissue_regions']}
• Total Tissue Area: {summary['workflow2_summary']['total_tissue_area']} pixels
• Activity Density: {summary['workflow2_summary']['activity_density']:.3f}

INTEGRATED ANALYSIS
------------------
• Overall Quality: {summary['integrated_analysis']['analysis_quality'].upper()}
• Total Activity: {summary['integrated_analysis']['total_activity']:.2f}

KEY FINDINGS
-----------
"""
        for finding in summary['integrated_analysis']['key_findings']:
            summary_text += f"• {finding}\n"
        
        summary_text += f"""
RECOMMENDATIONS
--------------
"""
        for rec in summary['quality_assessment']['recommendations']:
            summary_text += f"• {rec}\n"
        
        summary_text += f"""
OUTPUT FILES
-----------
• Aligned iQID Stack: {summary['file_outputs']['aligned_iqid_stack']}
• Tissue Activity Data: {summary['file_outputs']['tissue_activity_data']}
• Quantitative Analysis: {summary['file_outputs']['quantitative_analysis']}
• Visualizations: {summary['file_outputs']['visualizations_dir']}

For detailed technical information, see the comprehensive JSON report.
"""
        
        # Save human-readable summary
        summary_path = self.master_output_dir / "analysis_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        logger.info(f"Human-readable summary saved to {summary_path}")


def main():
    """Main execution function."""
    import numpy as np  # Import numpy for calculations
    
    print("🔬 UCSF Complete iQID-H&E Analysis Pipeline")
    print("=" * 60)
    print("This pipeline will run both workflows:")
    print("1. iQID Raw Data Alignment")
    print("2. H&E-iQID Co-registration and Analysis")
    print("=" * 60)
    
    # Initialize complete pipeline
    pipeline = UCSFCompletePipeline()
    
    # Get data paths from the workflow configs
    config1 = pipeline.workflow1.config
    config2 = pipeline.workflow2.config
    
    # Use data paths from config1 (should have UCSF structure)
    data_paths = config1.get("data_paths", {})
    
    # Check if we have UCSF data structure
    ucsf_base = data_paths.get("ucsf_base_dir")
    if ucsf_base:
        print(f"📂 UCSF base directory: {ucsf_base}")
        
        # Get iQID data path from ReUpload (has raw iQID data)
        if "reupload" in data_paths:
            reupload_data = data_paths["reupload"]
            iqid_reupload = reupload_data.get("iqid_reupload", {})
            
            # Try 3D scans first, then sequential
            if "3d_scans" in iqid_reupload:
                raw_iqid_path = iqid_reupload["3d_scans"]
                print(f"📊 iQID source: ReUpload 3D scans")
            elif "sequential_scans" in iqid_reupload:
                raw_iqid_path = iqid_reupload["sequential_scans"]
                print(f"📊 iQID source: ReUpload sequential scans")
            else:
                raw_iqid_path = reupload_data.get("base_path", "data/raw_iqid")
                print(f"📊 iQID source: ReUpload base path")
        
        # Fallback to DataPush1 if ReUpload not available
        elif "datapush1" in data_paths:
            datapush1_data = data_paths["datapush1"]
            iqid_images = datapush1_data.get("iqid_images", {})
            
            if "3d_scans" in iqid_images:
                raw_iqid_path = iqid_images["3d_scans"]
                print(f"📊 iQID source: DataPush1 3D scans")
            elif "sequential_10um" in iqid_images:
                raw_iqid_path = iqid_images["sequential_10um"]
                print(f"📊 iQID source: DataPush1 sequential sections")
            else:
                raw_iqid_path = datapush1_data.get("base_path", "data/raw_iqid")
                print(f"📊 iQID source: DataPush1 base path")
        else:
            raw_iqid_path = ucsf_base
            print(f"📊 iQID source: UCSF base directory")
            
        # Get H&E data path from DataPush1 (has H&E images)
        if "datapush1" in data_paths:
            datapush1_data = data_paths["datapush1"]
            he_images = datapush1_data.get("he_images", {})
            
            if "3d_scans" in he_images:
                he_images_path = he_images["3d_scans"]
                print(f"📊 H&E source: DataPush1 3D scans")
            elif "sequential_10um" in he_images:
                he_images_path = he_images["sequential_10um"]
                print(f"📊 H&E source: DataPush1 sequential sections")
            else:
                he_images_path = datapush1_data.get("base_path", "data/he_histology")
                print(f"📊 H&E source: DataPush1 base path")
        else:
            he_images_path = ucsf_base
            print(f"📊 H&E source: UCSF base directory")
    else:
        # Fallback to default paths
        raw_iqid_path = "data/raw_iqid"
        he_images_path = "data/he_histology"
        print(f"📂 Using default data paths")
    
    print(f"📂 iQID data location: {raw_iqid_path}")
    print(f"📂 H&E data location: {he_images_path}")
    
    # Check readonly policy
    readonly_warning = data_paths.get("readonly_warning")
    if readonly_warning:
        print(f"⚠️  {readonly_warning}")
        print(f"� All outputs will be saved to local directories")
    print(f"📂 H&E data location: {he_images_path}")
    
    # Check readonly policy
    storage_policy = config1.get("storage_policy", {})
    if storage_policy.get("enforce_readonly", False):
        print(f"⚠️  READONLY mode enforced - source data will not be modified")
        output_base = storage_policy.get("output_base", ".")
        print(f"📁 All outputs will be saved to: {output_base}")
    
    # Run complete analysis
    try:
        results = pipeline.run_complete_analysis(raw_iqid_path, he_images_path)
        
        print("\n" + "=" * 60)
        print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("=" * 60)
        
        # Print summary
        summary = results['comprehensive_summary']
        print(f"📊 Processing Time: {summary['pipeline_metadata']['total_processing_time']:.1f} seconds")
        print(f"📁 Master Output Directory: outputs/complete_analysis/")
        print(f"📈 Analysis Quality: {summary['integrated_analysis']['analysis_quality'].upper()}")
        
        # Print key results
        w1_summary = summary['workflow1_summary']
        w2_summary = summary['workflow2_summary']
        
        print(f"\n🔬 iQID Alignment Results:")
        print(f"   • {w1_summary['n_frames_processed']} frames aligned")
        print(f"   • Mean correlation: {w1_summary['mean_correlation']:.3f}")
        
        print(f"\n🧬 H&E Co-registration Results:")
        print(f"   • {w2_summary['n_tissue_regions']} tissue regions found")
        print(f"   • Registration correlation: {w2_summary['registration_correlation']:.3f}")
        print(f"   • Activity density: {w2_summary['activity_density']:.3f}")
        
        print(f"\n📄 Reports Generated:")
        print(f"   • Comprehensive JSON: outputs/complete_analysis/comprehensive_analysis_report.json")
        print(f"   • Human-readable summary: outputs/complete_analysis/analysis_summary.txt")
        print(f"   • CSV data: outputs/he_iqid_analysis/tissue_activity_data.csv")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logger.error(f"Complete pipeline failed: {e}")
        raise


if __name__ == "__main__":
    results = main()
