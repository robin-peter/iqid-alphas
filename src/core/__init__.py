"""
Core processing module for iQID data.
"""

# Import core components with error handling for missing dependencies
try:
    from .iqid.process_object import ClusterData
    from .iqid.align import assemble_stack, assemble_stack_hne, coarse_stack, pad_stack_he, crop_down
    from .iqid.helper import *
    from .iqid.dpk import *
    _core_imports_successful = True
except ImportError as e:
    print(f"Warning: Some core modules could not be imported: {e}")
    _core_imports_successful = False

# Import spec separately as it has additional dependencies
try:
    from .iqid.spec import *
    _spec_import_successful = True
except ImportError as e:
    print(f"Warning: Spec module could not be imported (missing becquerel dependency): {e}")
    _spec_import_successful = False

__all__ = [
    'ClusterData',
    'assemble_stack', 'assemble_stack_hne', 'coarse_stack', 'pad_stack_he', 'crop_down'
] if _core_imports_successful else []
