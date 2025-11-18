#!/usr/bin/env python3
"""Quick test for List_Schedule platform loading"""
import sys
import os

# Change to src directory so relative imports work
os.chdir('src')
sys.path.insert(0, '.')

import config
import List_Schedule as ls

# Test different applications
test_cases = [
    'T2_var_094.json',
    'T20.json',
    'T5_var_001.json'
]

print("="*70)
print("Testing List_Schedule.read_platform_model()")
print("="*70)

for app in test_cases:
    config.file_name = app
    try:
        platform = ls.read_platform_model()
        print(f"✅ {app:20} -> Platform loaded ({len(platform['nodes'])} nodes)")
    except Exception as e:
        print(f"❌ {app:20} -> ERROR: {e}")

print("="*70)
os.chdir('..')  # Change back
