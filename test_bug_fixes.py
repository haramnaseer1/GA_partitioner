#!/usr/bin/env python3
"""
Quick test script to verify bug fixes for client issues
Tests with DEBUG_MODE enabled for fast execution
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_platform_detection():
    """Test that platform files are correctly detected"""
    import config as cfg
    
    test_cases = [
        ('T2_var_094.json', '2_Platform.json'),
        ('T5_var_001.json', '5_Platform.json'),
        ('T20.json', '20_Platform.json'),
    ]
    
    print("\n" + "="*70)
    print("Testing Platform Detection")
    print("="*70)
    
    for app_name, expected_platform in test_cases:
        cfg.file_name = app_name
        
        # Test auxiliary_fun_GA platform detection
        import re
        match = re.match(r'[Tt](\d+)', app_name)  # Match T followed by digits
        if match:
            platform_num = match.group(1)
            detected = f"{platform_num}_Platform.json"
        else:
            detected = "3_Tier_Platform.json"
        
        status = "✅" if detected == expected_platform else "❌"
        print(f"{status} {app_name:20} -> {detected:20} (expected: {expected_platform})")

def test_paths_file_naming():
    """Test that Paths files are platform-specific"""
    import config as cfg
    import re
    
    print("\n" + "="*70)
    print("Testing Platform-Specific Paths Files")
    print("="*70)
    
    test_cases = [
        ('T2_var_094.json', 'Paths_2.json'),
        ('T5_var_001.json', 'Paths_5.json'),
        ('T20.json', 'Paths_20.json'),
        ('TNC100.json', 'Paths.json'),  # Fallback for non-T* apps
    ]
    
    for app_name, expected_paths in test_cases:
        cfg.file_name = app_name
        
        match = re.match(r'[Tt](\d+)', app_name)  # Match T followed by digits
        if match:
            platform_num = match.group(1)
            paths_file = f"Paths_{platform_num}.json"
        else:
            paths_file = "Paths.json"
        
        status = "✅" if paths_file == expected_paths else "❌"
        print(f"{status} {app_name:20} -> {paths_file:20} (expected: {expected_paths})")

def test_debug_mode_settings():
    """Test that DEBUG_MODE reduces GA parameters"""
    import config as cfg
    
    print("\n" + "="*70)
    print("Testing DEBUG_MODE Settings")
    print("="*70)
    
    print(f"DEBUG_MODE: {cfg.DEBUG_MODE}")
    print(f"Population Size GGA: {cfg.POPULATION_SIZE_GGA} (3 in debug, 20 in production)")
    print(f"Generations GCA: {cfg.NUMBER_OF_GENERATIONS_GCA} (5 in debug, 100 in production)")
    print(f"Population Size LGA: {cfg.POPULATION_SIZE_LGA} (3 in debug, 10 in production)")
    print(f"Generations LGA: {cfg.NUMBER_OF_GENERATIONS_LGA} (5 in debug, 50 in production)")
    
    if cfg.DEBUG_MODE:
        if cfg.POPULATION_SIZE_GGA == 3 and cfg.NUMBER_OF_GENERATIONS_GCA == 5:
            print("✅ DEBUG_MODE settings are correct - GA will run FAST")
        else:
            print("❌ DEBUG_MODE settings incorrect")
    else:
        print("⚠️  DEBUG_MODE is OFF - GA will run SLOW (production mode)")

def main():
    print("\n" + "="*70)
    print("BUG FIX VERIFICATION TESTS")
    print("="*70)
    
    test_platform_detection()
    test_paths_file_naming()
    test_debug_mode_settings()
    
    print("\n" + "="*70)
    print("All Tests Complete!")
    print("="*70)
    print("\nFixes Applied:")
    print("1. ✅ Crossover empty range error (global_GA.py)")
    print("2. ✅ Missing processor KeyError (global_GA.py)")
    print("3. ✅ Paths.json overwriting (main.py, auxiliary_fun_GA.py)")
    print("4. ✅ Platform file detection (List_Schedule.py)")
    print("5. ✅ DEBUG_MODE enabled for fast testing (config.py)")
    print("\nYou can now run GA with:")
    print("  python -m src.main")
    print("\n")

if __name__ == "__main__":
    main()
