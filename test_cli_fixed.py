#!/usr/bin/env python3
"""
Simple test to check CLI functionality after fixes
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_import():
    """Test if we can import the CLI module."""
    try:
        from iqid_alphas.cli import IQIDCLIProcessor
        print("✅ CLI import successful")
        return True
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_discovery():
    """Test data discovery."""
    try:
        from iqid_alphas.cli import IQIDCLIProcessor
        
        processor = IQIDCLIProcessor()
        print("✅ CLI processor created")
        
        # Test with a small path
        test_path = "data/DataPush1/iQID"
        if Path(test_path).exists():
            discovered = processor.discover_data(test_path)
            print(f"✅ Discovery successful: {len(discovered.get('iqid_samples', []))} iQID samples")
            return True
        else:
            print(f"⚠️  Test path {test_path} not found")
            return False
            
    except Exception as e:
        print(f"❌ Discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🧪 CLI Post-Fix Test")
    print("=" * 30)
    
    import_ok = test_import()
    
    if import_ok:
        discovery_ok = test_discovery()
    else:
        discovery_ok = False
    
    print("\n" + "=" * 30)
    print(f"Import: {'✅' if import_ok else '❌'}")
    print(f"Discovery: {'✅' if discovery_ok else '❌'}")
    
    if import_ok and discovery_ok:
        print("\n🎉 CLI is working correctly with sample-based processing!")
    else:
        print("\n⚠️  Some issues remain. Check errors above.")
