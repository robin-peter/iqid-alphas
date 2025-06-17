#!/usr/bin/env python3
"""
Complete repository organization script to move/remove old files from root
and ensure proper directory structure.
"""

import os
import shutil
from pathlib import Path

def main():
    repo_root = Path("/home/wxc151/iqid-alphas")
    
    print("=== REPOSITORY ORGANIZATION ===")
    print(f"Working in: {repo_root}")
    
    # Files that should be moved from root to proper locations
    files_to_move = {
        # Old automation scripts → archive
        "automate_dose_kernel_processing.py": "archive/",
        "automate_image_alignment.py": "archive/",
        "automate_processing.py": "archive/",
        "check_dependencies.py": "archive/",
        "config.json": "archive/",
        
        # Test files → archive (since we have new tests in tests/)
        "test_automate_dose_kernel_processing.py": "archive/",
        "test_automate_image_alignment.py": "archive/",
        "test_automate_processing.py": "archive/",
        
        # Documentation files → docs/
        "design_document.md": "docs/",
        "iqid_align_documentation.md": "docs/",
        "iqid_helper_documentation.md": "docs/",
        "iqid_process_object_documentation.md": "docs/",
        "ucsf_ac225_iqid_processing_plan.md": "docs/",
    }
    
    # Directories that should be moved/removed
    dirs_to_move = {
        # Old directories → archive
        "demo_notebooks": "archive/",
        "misc_notebooks": "archive/",
        "iqid": "archive/",  # Old iqid dir (we have src/core/iqid/ now)
    }
    
    print("\n1. Moving files to proper locations...")
    for file_name, target_dir in files_to_move.items():
        source = repo_root / file_name
        target_dir_path = repo_root / target_dir
        target_file = target_dir_path / file_name
        
        if source.exists():
            target_dir_path.mkdir(parents=True, exist_ok=True)
            if not target_file.exists():
                shutil.move(str(source), str(target_file))
                print(f"   Moved: {file_name} → {target_dir}")
            else:
                print(f"   Exists: {target_dir}{file_name} (removing duplicate)")
                source.unlink()
        else:
            print(f"   Missing: {file_name} (already moved)")
    
    print("\n2. Moving directories to proper locations...")
    for dir_name, target_dir in dirs_to_move.items():
        source = repo_root / dir_name
        target_dir_path = repo_root / target_dir
        target_subdir = target_dir_path / dir_name
        
        if source.exists() and source.is_dir():
            target_dir_path.mkdir(parents=True, exist_ok=True)
            if not target_subdir.exists():
                shutil.move(str(source), str(target_subdir))
                print(f"   Moved: {dir_name}/ → {target_dir}")
            else:
                print(f"   Exists: {target_dir}{dir_name}/ (removing duplicate)")
                shutil.rmtree(source)
        else:
            print(f"   Missing: {dir_name}/ (already moved)")
    
    print("\n3. Verifying organized structure...")
    expected_dirs = [
        "src/",
        "pipelines/", 
        "scripts/",
        "docs/",
        "archive/",
    ]
    
    for expected_dir in expected_dirs:
        dir_path = repo_root / expected_dir
        status = "✓" if dir_path.exists() else "✗"
        print(f"   {status} {expected_dir}")
    
    print("\n4. Final root directory contents:")
    root_items = []
    for item in repo_root.iterdir():
        if not item.name.startswith('.') and item.name not in ['outputs']:
            item_type = "dir" if item.is_dir() else "file"
            root_items.append(f"   - {item.name} ({item_type})")
    
    for item in sorted(root_items):
        print(item)
    
    print("\n=== ORGANIZATION COMPLETE ===")
    print("Repository now has clean, professional structure!")

if __name__ == "__main__":
    main()
