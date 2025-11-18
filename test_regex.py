import re
import os

# Simulate the detection logic from auxiliary_fun_GA.py
app_name = "T2_var_003.json"

match = re.match(r'[Tt](\d+)_', app_name)
print(f"App name: {app_name}")
print(f"Regex match: {match}")

if match:
    platform_num = match.group(1)
    print(f"Extracted platform number: {platform_num}")
    pltfile = f"../Platform/{platform_num}_Platform.json"
    print(f"Platform file: {pltfile}")
else:
    print("No match - would use fallback (Platform 5)")
    
# Check if file exists
if match:
    platform_path = f"Platform/{platform_num}_Platform.json"
    exists = os.path.exists(platform_path)
    print(f"Platform file exists: {exists}")
