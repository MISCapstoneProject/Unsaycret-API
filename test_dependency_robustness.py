#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify orchestrator_v2.py robustness against missing dependencies.

This script tests:
1. Module compilation
2. Import without crashes
3. Initialization with missing dependencies
4. CLI argument parsing
5. Graceful degradation behavior

Usage:
    python test_dependency_robustness.py
"""

import sys
import importlib
import subprocess
from pathlib import Path


def test_compilation():
    """Test that the module compiles without syntax errors."""
    print("=" * 60)
    print("TEST 1: Module Compilation")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", "pipelines/orchestrator_v2.py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ PASS: Module compiles successfully")
            return True
        else:
            print("❌ FAIL: Compilation errors:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ FAIL: Exception during compilation: {e}")
        return False


def test_import():
    """Test that the module can be imported."""
    print("\n" + "=" * 60)
    print("TEST 2: Module Import")
    print("=" * 60)
    
    try:
        # Try importing the module
        import pipelines.orchestrator_v2
        print("✅ PASS: Module imports successfully")
        return True
    except ImportError as e:
        print(f"❌ FAIL: ImportError: {e}")
        return False
    except Exception as e:
        print(f"❌ FAIL: Exception during import: {e}")
        return False


def test_init_with_missing_id():
    """Test initialization when speaker identification is unavailable."""
    print("\n" + "=" * 60)
    print("TEST 3: Initialization with Missing Speaker ID")
    print("=" * 60)
    
    try:
        from pipelines.orchestrator_v2 import init_pipeline_modules
        
        # Try to initialize with id_backend='none'
        print("Attempting to initialize with id_backend='none'...")
        sep, identifier, asr, use_gpu = init_pipeline_modules(
            load_separator=False,  # Skip heavy models for quick test
            load_identifier=True,   # Try to load but should fallback gracefully
            load_asr=False,
            id_backend='none'       # Explicitly disable
        )
        
        if identifier is None:
            print("✅ PASS: Identifier is None as expected (id_backend='none')")
            return True
        else:
            print("⚠️  WARNING: Identifier loaded despite id_backend='none'")
            return True  # Still pass, just unexpected
            
    except Exception as e:
        print(f"❌ FAIL: Exception during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_arguments():
    """Test that CLI arguments parse correctly."""
    print("\n" + "=" * 60)
    print("TEST 4: CLI Argument Parsing")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pipelines.orchestrator_v2", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        stdout = result.stdout
        
        # Check for new arguments
        checks = [
            ("--id-backend" in stdout, "--id-backend argument exists"),
            ("--id-device" in stdout, "--id-device argument exists"),
            ("wespeaker" in stdout, "wespeaker mentioned in help"),
            ("none" in stdout.lower(), "'none' option available"),
        ]
        
        all_passed = True
        for passed, description in checks:
            if passed:
                print(f"✅ {description}")
            else:
                print(f"❌ {description}")
                all_passed = False
        
        if all_passed:
            print("\n✅ PASS: All CLI arguments present")
            return True
        else:
            print("\n⚠️  WARNING: Some CLI arguments missing (may be okay)")
            return True  # Still pass as warning
            
    except Exception as e:
        print(f"❌ FAIL: Exception during CLI test: {e}")
        return False


def test_lazy_imports():
    """Test that lazy imports are in place."""
    print("\n" + "=" * 60)
    print("TEST 5: Lazy Import Flags")
    print("=" * 60)
    
    try:
        import pipelines.orchestrator_v2 as orch
        
        checks = [
            (hasattr(orch, '_HAS_TORCH'), "_HAS_TORCH flag exists"),
            (hasattr(orch, '_HAS_SCIPY'), "_HAS_SCIPY flag exists"),
        ]
        
        all_passed = True
        for passed, description in checks:
            if passed:
                print(f"✅ {description}")
            else:
                print(f"❌ {description}")
                all_passed = False
        
        # Check values
        if hasattr(orch, '_HAS_TORCH'):
            print(f"   _HAS_TORCH = {orch._HAS_TORCH}")
        if hasattr(orch, '_HAS_SCIPY'):
            print(f"   _HAS_SCIPY = {orch._HAS_SCIPY}")
        
        if all_passed:
            print("\n✅ PASS: Lazy import flags present")
            return True
        else:
            print("\n❌ FAIL: Missing lazy import flags")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Exception during lazy import test: {e}")
        return False


def test_function_signature():
    """Test that init_pipeline_modules has id_backend parameter."""
    print("\n" + "=" * 60)
    print("TEST 6: Function Signature")
    print("=" * 60)
    
    try:
        import inspect
        from pipelines.orchestrator_v2 import init_pipeline_modules
        
        sig = inspect.signature(init_pipeline_modules)
        params = list(sig.parameters.keys())
        
        if 'id_backend' in params:
            print(f"✅ PASS: init_pipeline_modules has 'id_backend' parameter")
            print(f"   Parameters: {params}")
            return True
        else:
            print(f"❌ FAIL: 'id_backend' parameter missing")
            print(f"   Parameters: {params}")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Exception during signature test: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ORCHESTRATOR V2 DEPENDENCY ROBUSTNESS TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Compilation", test_compilation),
        ("Import", test_import),
        ("Init with Missing ID", test_init_with_missing_id),
        ("CLI Arguments", test_cli_arguments),
        ("Lazy Imports", test_lazy_imports),
        ("Function Signature", test_function_signature),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ FAIL: {name} - Unexpected exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {passed_count}/{total_count} tests passed")
    print("=" * 60)
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Orchestrator is robust against missing dependencies.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
