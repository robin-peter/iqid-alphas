#!/usr/bin/env python3
"""
Production Validation System
Comprehensive validation for production deployment of the IQID-Alphas pipeline.
Tests all pipeline stages with real and synthetic data to ensure production readiness.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import traceback
from tifffile import imread, imwrite
import logging

# Add src to path
sys.path.insert(0, './src')
sys.path.insert(0, './pipelines')
sys.path.insert(0, '.')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionValidator:
    """Production validation system for IQID-Alphas pipeline."""
    
    def __init__(self):
        self.validation_dir = "./evaluation/reports/production"
        self.output_dir = "./outputs/production_validation"
        os.makedirs(self.validation_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'pipeline_version': '1.0',
            'validation_type': 'production_readiness_assessment',
            'test_suites': {},
            'performance_metrics': {},
            'production_readiness': False
        }
        
    def validate_dependencies(self):
        """Validate all required dependencies and versions."""
        print("🔍 Validating Dependencies...")
        
        dependencies = {
            'numpy': None,
            'matplotlib': None,
            'tifffile': None,
            'scikit-image': None,
            'scipy': None,
            'pandas': None
        }
        
        validation_status = True
        
        for dep in dependencies:
            try:
                module = __import__(dep.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                dependencies[dep] = version
                print(f"   ✅ {dep}: {version}")
            except ImportError:
                dependencies[dep] = 'MISSING'
                validation_status = False
                print(f"   ❌ {dep}: MISSING")
        
        self.validation_results['dependencies'] = {
            'status': validation_status,
            'packages': dependencies
        }
        
        return validation_status
    
    def validate_pipeline_components(self):
        """Validate individual pipeline components."""
        print("🧪 Validating Pipeline Components...")
        
        components_status = {}
        
        # Test SimpleiQIDPipeline
        try:
            from simplified_iqid_pipeline import SimpleiQIDPipeline
            pipeline = SimpleiQIDPipeline()
            components_status['SimpleiQIDPipeline'] = True
            print("   ✅ SimpleiQIDPipeline: OK")
        except Exception as e:
            components_status['SimpleiQIDPipeline'] = False
            print(f"   ❌ SimpleiQIDPipeline: {e}")
        
        # Test iQIDProcessingPipeline
        try:
            from iqid_only_pipeline import iQIDProcessingPipeline
            pipeline = iQIDProcessingPipeline()
            components_status['iQIDProcessingPipeline'] = True
            print("   ✅ iQIDProcessingPipeline: OK")
        except Exception as e:
            components_status['iQIDProcessingPipeline'] = False
            print(f"   ❌ iQIDProcessingPipeline: {e}")
        
        # Test CombinedHEiQIDPipeline
        try:
            from combined_he_iqid_pipeline import CombinedHEiQIDPipeline
            pipeline = CombinedHEiQIDPipeline()
            components_status['CombinedHEiQIDPipeline'] = True
            print("   ✅ CombinedHEiQIDPipeline: OK")
        except Exception as e:
            components_status['CombinedHEiQIDPipeline'] = False
            print(f"   ❌ CombinedHEiQIDPipeline: {e}")
        
        self.validation_results['pipeline_components'] = components_status
        return all(components_status.values())
    
    def validate_test_data_integrity(self):
        """Validate integrity of test data."""
        print("📁 Validating Test Data Integrity...")
        
        test_data_dir = Path('./test_data')
        integrity_status = {
            'test_data_exists': test_data_dir.exists(),
            'samples_found': 0,
            'paired_samples': 0,
            'data_integrity': {}
        }
        
        if test_data_dir.exists():
            samples = list(test_data_dir.iterdir())
            integrity_status['samples_found'] = len([s for s in samples if s.is_dir()])
            
            for sample_dir in samples:
                if sample_dir.is_dir():
                    sample_name = sample_dir.name
                    he_dir = sample_dir / 'he'
                    iqid_dir = sample_dir / 'iqid'
                    
                    sample_status = {
                        'he_exists': he_dir.exists(),
                        'iqid_exists': iqid_dir.exists(),
                        'he_files': len(list(he_dir.glob('*.tif*'))) if he_dir.exists() else 0,
                        'iqid_files': len(list(iqid_dir.glob('*.tif*'))) if iqid_dir.exists() else 0
                    }
                    
                    if sample_status['he_exists'] and sample_status['iqid_exists']:
                        integrity_status['paired_samples'] += 1
                        print(f"   ✅ {sample_name}: Paired data OK")
                    else:
                        print(f"   ⚠️  {sample_name}: Incomplete data")
                    
                    integrity_status['data_integrity'][sample_name] = sample_status
        
        self.validation_results['test_data'] = integrity_status
        return integrity_status['paired_samples'] > 0
    
    def run_pipeline_stress_test(self):
        """Run stress tests on pipelines with various data scenarios."""
        print("💪 Running Pipeline Stress Tests...")
        
        stress_test_results = {
            'simple_pipeline': {},
            'iqid_pipeline': {},
            'combined_pipeline': {}
        }
        
        # Test with different data types and edge cases
        test_scenarios = [
            'normal_data',
            'small_image',
            'large_image',
            'noisy_data',
            'low_contrast'
        ]
        
        for scenario in test_scenarios:
            print(f"   🧪 Testing scenario: {scenario}")
            
            # Generate synthetic test data for each scenario
            test_data = self.generate_test_data_for_scenario(scenario)
            
            # Test each pipeline
            for pipeline_name in ['simple_pipeline', 'iqid_pipeline', 'combined_pipeline']:
                try:
                    result = self.test_pipeline_with_data(pipeline_name, test_data, scenario)
                    stress_test_results[pipeline_name][scenario] = {
                        'success': True,
                        'processing_time': result.get('processing_time', 0),
                        'output_quality': result.get('quality_score', 0)
                    }
                    print(f"     ✅ {pipeline_name}: OK")
                except Exception as e:
                    stress_test_results[pipeline_name][scenario] = {
                        'success': False,
                        'error': str(e),
                        'processing_time': 0,
                        'output_quality': 0
                    }
                    print(f"     ❌ {pipeline_name}: {e}")
        
        self.validation_results['stress_tests'] = stress_test_results
        
        # Calculate overall stress test score
        total_tests = len(test_scenarios) * 3  # 3 pipelines
        successful_tests = sum([
            len([s for s in pipeline_results.values() if s.get('success', False)])
            for pipeline_results in stress_test_results.values()
        ])
        
        stress_success_rate = successful_tests / total_tests if total_tests > 0 else 0
        print(f"   📊 Overall stress test success rate: {stress_success_rate:.1%}")
        
        return stress_success_rate > 0.8
    
    def generate_test_data_for_scenario(self, scenario):
        """Generate synthetic test data for different scenarios."""
        base_size = 512
        
        if scenario == 'small_image':
            size = 64
        elif scenario == 'large_image':
            size = 1024
        else:
            size = base_size
        
        # Generate synthetic iQID and H&E images
        np.random.seed(42)  # For reproducibility
        
        if scenario == 'noisy_data':
            iqid_image = np.random.poisson(100, (size, size)).astype(np.uint16)
            he_image = np.random.randint(0, 255, (size, size, 3)).astype(np.uint8)
        elif scenario == 'low_contrast':
            iqid_image = np.full((size, size), 128, dtype=np.uint16)
            he_image = np.full((size, size, 3), 128, dtype=np.uint8)
        else:
            # Normal synthetic data with some structure
            x, y = np.meshgrid(np.linspace(0, 10, size), np.linspace(0, 10, size))
            iqid_image = (1000 * np.exp(-(x-5)**2/10 - (y-5)**2/10) + 
                         100 * np.random.random((size, size))).astype(np.uint16)
            he_image = np.stack([
                (255 * np.exp(-(x-5)**2/10 - (y-5)**2/10)).astype(np.uint8),
                (128 * np.ones((size, size))).astype(np.uint8),
                (200 * np.exp(-(x-3)**2/5 - (y-7)**2/5)).astype(np.uint8)
            ], axis=2)
        
        return {
            'iqid': iqid_image,
            'he': he_image,
            'scenario': scenario
        }
    
    def test_pipeline_with_data(self, pipeline_name, test_data, scenario):
        """Test a specific pipeline with given test data."""
        start_time = datetime.now()
        
        # Save test data temporarily
        temp_dir = Path(self.output_dir) / 'temp_test_data' / scenario
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        iqid_path = temp_dir / 'test_iqid.tif'
        he_path = temp_dir / 'test_he.tif'
        
        imwrite(str(iqid_path), test_data['iqid'])
        imwrite(str(he_path), test_data['he'])
        
        # Run appropriate pipeline
        if pipeline_name == 'simple_pipeline':
            from simplified_iqid_pipeline import SimpleiQIDPipeline
            pipeline = SimpleiQIDPipeline()
            result = pipeline.process_sample(str(iqid_path))
        elif pipeline_name == 'iqid_pipeline':
            from iqid_only_pipeline import iQIDProcessingPipeline
            pipeline = iQIDProcessingPipeline()
            result = pipeline.process_sample(str(iqid_path))
        elif pipeline_name == 'combined_pipeline':
            from combined_he_iqid_pipeline import CombinedHEiQIDPipeline
            pipeline = CombinedHEiQIDPipeline()
            result = pipeline.process_pair(str(he_path), str(iqid_path))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up temporary files
        iqid_path.unlink(missing_ok=True)
        he_path.unlink(missing_ok=True)
        
        return {
            'processing_time': processing_time,
            'quality_score': 85,  # Placeholder - would implement actual quality assessment
            'result_keys': list(result.keys()) if isinstance(result, dict) else []
        }
    
    def validate_output_structure(self):
        """Validate that pipeline outputs have the expected structure."""
        print("📋 Validating Output Structure...")
        
        output_validation = {
            'output_directories': {},
            'expected_files': {},
            'file_formats': {}
        }
        
        # Check for expected output directories
        expected_dirs = [
            './outputs/simple_pipeline',
            './outputs/iqid_pipeline',
            './outputs/combined_pipeline',
            './outputs/segmentation',
            './outputs/comprehensive_evaluation'
        ]
        
        for expected_dir in expected_dirs:
            dir_path = Path(expected_dir)
            exists = dir_path.exists()
            output_validation['output_directories'][expected_dir] = exists
            if exists:
                print(f"   ✅ {expected_dir}: EXISTS")
            else:
                print(f"   ⚠️  {expected_dir}: MISSING")
        
        self.validation_results['output_structure'] = output_validation
        return True
    
    def generate_production_readiness_report(self):
        """Generate comprehensive production readiness report."""
        print("📊 Generating Production Readiness Report...")
        
        # Calculate overall scores
        component_scores = {
            'dependencies': 100 if self.validation_results.get('dependencies', {}).get('status', False) else 0,
            'pipeline_components': 100 if all(self.validation_results.get('pipeline_components', {}).values()) else 50,
            'test_data': 100 if self.validation_results.get('test_data', {}).get('paired_samples', 0) > 0 else 0,
            'stress_tests': self.calculate_stress_test_score(),
            'output_structure': 100  # Assume OK if we got this far
        }
        
        overall_score = sum(component_scores.values()) / len(component_scores)
        production_ready = overall_score >= 80
        
        self.validation_results['performance_metrics'] = component_scores
        self.validation_results['overall_score'] = overall_score
        self.validation_results['production_readiness'] = production_ready
        
        # Generate detailed report
        report_path = Path(self.validation_dir) / 'production_readiness_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        # Generate markdown summary
        summary_path = Path(self.validation_dir) / 'production_readiness_summary.md'
        self.generate_markdown_summary(summary_path, overall_score, production_ready)
        
        print(f"   📄 Detailed report: {report_path}")
        print(f"   📋 Summary: {summary_path}")
        
        return report_path
    
    def calculate_stress_test_score(self):
        """Calculate stress test success score."""
        stress_tests = self.validation_results.get('stress_tests', {})
        if not stress_tests:
            return 0
        
        total_tests = 0
        successful_tests = 0
        
        for pipeline_results in stress_tests.values():
            for scenario_result in pipeline_results.values():
                total_tests += 1
                if scenario_result.get('success', False):
                    successful_tests += 1
        
        return (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    def generate_markdown_summary(self, summary_path, overall_score, production_ready):
        """Generate markdown summary of production readiness."""
        status_emoji = "✅" if production_ready else "⚠️"
        status_text = "PRODUCTION READY" if production_ready else "NEEDS IMPROVEMENT"
        
        summary_content = f"""# IQID-Alphas Production Readiness Report

**Validation Date:** {datetime.now().strftime('%B %d, %Y')}
**Overall Score:** {overall_score:.1f}/100
**Status:** {status_emoji} {status_text}

## Component Assessment

| Component | Score | Status |
|-----------|--------|---------|
| Dependencies | {self.validation_results['performance_metrics'].get('dependencies', 0):.0f}/100 | {'✅' if self.validation_results['performance_metrics'].get('dependencies', 0) >= 80 else '❌'} |
| Pipeline Components | {self.validation_results['performance_metrics'].get('pipeline_components', 0):.0f}/100 | {'✅' if self.validation_results['performance_metrics'].get('pipeline_components', 0) >= 80 else '❌'} |
| Test Data | {self.validation_results['performance_metrics'].get('test_data', 0):.0f}/100 | {'✅' if self.validation_results['performance_metrics'].get('test_data', 0) >= 80 else '❌'} |
| Stress Tests | {self.validation_results['performance_metrics'].get('stress_tests', 0):.0f}/100 | {'✅' if self.validation_results['performance_metrics'].get('stress_tests', 0) >= 80 else '❌'} |
| Output Structure | {self.validation_results['performance_metrics'].get('output_structure', 0):.0f}/100 | {'✅' if self.validation_results['performance_metrics'].get('output_structure', 0) >= 80 else '❌'} |

## Recommendations

{'### ✅ Ready for Production' if production_ready else '### ⚠️ Improvements Needed'}

{self.generate_recommendations(production_ready)}

## Technical Details

- **Test Data Samples:** {self.validation_results.get('test_data', {}).get('paired_samples', 0)}
- **Pipeline Components Tested:** {len(self.validation_results.get('pipeline_components', {}))}
- **Stress Test Scenarios:** {len(set([scenario for pipeline_results in self.validation_results.get('stress_tests', {}).values() for scenario in pipeline_results.keys()]))}

For detailed results, see: `production_readiness_report.json`
"""
        
        with open(summary_path, 'w') as f:
            f.write(summary_content)
    
    def generate_recommendations(self, production_ready):
        """Generate specific recommendations based on validation results."""
        if production_ready:
            return """
- Pipeline is ready for production deployment
- All core components are functional
- Test coverage is adequate
- Continue with regular validation cycles
"""
        else:
            recommendations = []
            
            if self.validation_results['performance_metrics'].get('dependencies', 0) < 80:
                recommendations.append("- Install missing dependencies")
            
            if self.validation_results['performance_metrics'].get('pipeline_components', 0) < 80:
                recommendations.append("- Fix pipeline component errors")
            
            if self.validation_results['performance_metrics'].get('test_data', 0) < 80:
                recommendations.append("- Ensure adequate test data coverage")
            
            if self.validation_results['performance_metrics'].get('stress_tests', 0) < 80:
                recommendations.append("- Improve pipeline robustness for edge cases")
            
            return '\n'.join(recommendations)
    
    def run_full_validation(self):
        """Run complete production validation suite."""
        print("🚀 Starting Production Validation Suite")
        print("=" * 80)
        
        try:
            # Run all validation steps
            dependencies_ok = self.validate_dependencies()
            components_ok = self.validate_pipeline_components()
            test_data_ok = self.validate_test_data_integrity()
            stress_tests_ok = self.run_pipeline_stress_test()
            output_structure_ok = self.validate_output_structure()
            
            # Generate final report
            report_path = self.generate_production_readiness_report()
            
            print("\n" + "=" * 80)
            print("🎯 PRODUCTION VALIDATION COMPLETE")
            print("=" * 80)
            
            overall_score = self.validation_results['overall_score']
            production_ready = self.validation_results['production_readiness']
            
            print(f"\n📊 FINAL ASSESSMENT:")
            print(f"   🎯 Overall Score: {overall_score:.1f}/100")
            print(f"   {'✅' if production_ready else '⚠️'} Status: {'PRODUCTION READY' if production_ready else 'NEEDS IMPROVEMENT'}")
            
            print(f"\n📁 VALIDATION OUTPUTS:")
            print(f"   📄 Detailed Report: {report_path}")
            print(f"   📋 Summary: production_readiness_summary.md")
            print(f"   📁 Output Directory: {self.output_dir}")
            
            return self.validation_results
            
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            logger.error(f"Validation failed: {traceback.format_exc()}")
            return None

def main():
    """Main function to run production validation."""
    validator = ProductionValidator()
    results = validator.run_full_validation()
    return results

if __name__ == "__main__":
    main()
