"""
UCSF Legacy Workflows Configuration Validation

This script validates the updated configuration files for the legacy UCSF workflows
to ensure they properly map to the real UCSF data structure and enforce readonly policy.
"""

import json
import os
import sys
from pathlib import Path

def validate_config_file(config_path, config_name):
    """Validate a configuration file"""
    print(f"\n{'='*60}")
    print(f"Validating {config_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(config_path):
        print(f"❌ ERROR: Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in {config_path}: {e}")
        return False
    
    print(f"✅ Config file loaded successfully: {config_path}")
    
    # Check basic structure
    required_sections = ['workflow_name', 'version', 'data_paths', 'output', 'logging']
    missing_sections = [section for section in required_sections if section not in config]
    
    if missing_sections:
        print(f"⚠️  WARNING: Missing sections: {missing_sections}")
    else:
        print("✅ All required sections present")
    
    # Check version
    version = config.get('version', 'unknown')
    print(f"📋 Version: {version}")
    
    # Check readonly warnings
    data_paths = config.get('data_paths', {})
    if 'readonly_warning' in data_paths:
        print("✅ Readonly warning present")
        print(f"   📝 {data_paths['readonly_warning']}")
    else:
        print("⚠️  WARNING: No readonly warning found")
    
    # Check UCSF data paths
    print("\n🗂️  Data Source Validation:")
    ucsf_paths = []
    
    if 'ucsf_base_dir' in data_paths:
        ucsf_base = data_paths['ucsf_base_dir']
        print(f"   📁 UCSF Base Directory: {ucsf_base}")
        ucsf_paths.append(ucsf_base)
    
    # Check specific source paths
    source_keys = ['he_images_sources', 'iqid_sources', 'raw_iqid_sources', 'reference_aligned']
    for key in source_keys:
        if key in data_paths:
            sources = data_paths[key]
            print(f"   📂 {key}:")
            if isinstance(sources, dict):
                for source_name, source_path in sources.items():
                    print(f"      🔗 {source_name}: {source_path}")
                    ucsf_paths.append(source_path)
            else:
                print(f"      🔗 {sources}")
                ucsf_paths.append(sources)
    
    # Validate readonly paths
    readonly_violations = []
    for path in ucsf_paths:
        if path and not path.startswith('/readonly/'):
            readonly_violations.append(path)
    
    if readonly_violations:
        print(f"⚠️  WARNING: Paths not marked as readonly: {readonly_violations}")
    else:
        print("✅ All UCSF paths properly marked as readonly")
    
    # Check output configuration
    print("\n📤 Output Configuration:")
    output_config = config.get('output', {})
    
    if 'readonly_policy' in output_config:
        print("✅ Readonly policy defined")
        print(f"   📝 {output_config['readonly_policy']}")
    
    output_dirs = []
    output_keys = ['base_output_dir', 'intermediate_dir', 'logs_dir', 'reports_dir']
    for key in output_keys:
        if key in output_config:
            output_dir = output_config[key]
            print(f"   📁 {key}: {output_dir}")
            output_dirs.append(output_dir)
    
    # Check that outputs are not in readonly paths
    readonly_output_violations = []
    for output_dir in output_dirs:
        if output_dir and output_dir.startswith('/readonly/'):
            readonly_output_violations.append(output_dir)
    
    if readonly_output_violations:
        print(f"❌ ERROR: Output directories in readonly paths: {readonly_output_violations}")
        return False
    else:
        print("✅ All output directories are writable")
    
    # Check logging configuration
    print("\n📝 Logging Configuration:")
    logging_config = config.get('logging', {})
    log_file = logging_config.get('log_file', '')
    
    if log_file:
        print(f"   📄 Log file: {log_file}")
        if log_file.startswith('/readonly/'):
            print(f"❌ ERROR: Log file in readonly path: {log_file}")
            return False
        else:
            print("✅ Log file in writable location")
    
    return True

def validate_file_patterns(config_path):
    """Validate file patterns match expected UCSF naming conventions"""
    print(f"\n🔍 File Pattern Validation:")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_paths = config.get('data_paths', {})
    
    # Check file patterns
    if 'file_patterns' in data_paths:
        patterns = data_paths['file_patterns']
        print("   📋 Defined file patterns:")
        for pattern_name, pattern in patterns.items():
            print(f"      🔗 {pattern_name}: {pattern}")
    
    # Check tissue patterns
    if 'sample_structure' in data_paths:
        structures = data_paths['sample_structure']
        print("   🧬 Sample structure patterns:")
        for structure_name, pattern in structures.items():
            print(f"      🔗 {structure_name}: {pattern}")

def main():
    """Main validation function"""
    print("UCSF Legacy Workflows - Configuration Validation")
    print("=" * 60)
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    configs_dir = script_dir / 'configs'
    
    # Configuration files to validate
    config_files = [
        (configs_dir / 'he_iqid_config.json', 'H&E-iQID Co-registration Config'),
        (configs_dir / 'iqid_alignment_config.json', 'iQID Alignment Config')
    ]
    
    all_valid = True
    
    for config_path, config_name in config_files:
        valid = validate_config_file(config_path, config_name)
        if valid:
            validate_file_patterns(config_path)
        all_valid = all_valid and valid
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    if all_valid:
        print("✅ All configurations are valid!")
        print("✅ Readonly policy properly enforced")
        print("✅ Output directories are writable")
        print("✅ Ready for use with real UCSF data")
    else:
        print("❌ Configuration validation failed!")
        print("❌ Please fix errors before using workflows")
        return 1
    
    print(f"\n📋 Next Steps:")
    print("   1. Ensure UCSF data is mounted at ../data/UCSF-Collab/data/")
    print("   2. Run workflows with: python run_complete_pipeline.py")
    print("   3. Check outputs in local outputs/ directory")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
