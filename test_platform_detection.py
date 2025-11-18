"""
Test if GA now uses correct platform based on application filename.
"""
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

# Import config and modify file_name
from src import config as cfg

# Test platform detection for T2_var_003
cfg.file_name = "T2_var_003.json"
print(f"Testing with application: {cfg.file_name}")

# Simulate the platform detection logic from global_GA.py
import re
app_name = cfg.file_name
match = re.match(r'[Tt](\d+)_', app_name)
if match:
    platform_model_str = match.group(1)
else:
    platform_model_str = "5"
    if not os.path.exists(os.path.join(cfg.platform_dir_path, f"{platform_model_str}_Platform.json")):
        platform_model_str = "3"

print(f"Detected platform: {platform_model_str}")
print(f"Platform file: {cfg.platform_dir_path}/{platform_model_str}_Platform.json")

# Verify the file exists
platform_path = os.path.join(cfg.platform_dir_path, f"{platform_model_str}_Platform.json")
if os.path.exists(platform_path):
    print(f"✓ Platform file exists")
    
    # Load and check processor IDs
    import json
    with open(platform_path) as f:
        plat = json.load(f)
        proc_ids = sorted([n['id'] for n in plat['platform']['nodes'] if not n.get('is_router', False)])
        print(f"Platform {platform_model_str} processor IDs: {proc_ids}")
else:
    print(f"✗ Platform file NOT FOUND!")

print("\n" + "="*60)
print("Expected behavior:")
print("T2_var_003.json should use Platform 2 (processors 21-26)")
print("Solutions should now have processor IDs matching the correct platform")
