"""Quick test to verify message size fix on T2"""
import sys
import os

# Change to src directory for imports
os.chdir('src')
sys.path.insert(0, os.getcwd())

# Override config
import config
config.file_name = 'T2.json'

# Run global GA
from global_GA import global_ga_function

print("\n" + "="*70)
print("TESTING MESSAGE SIZE FIX")
print("="*70)
print(f"Application: {config.file_name}")
print(f"GGA Iterations: {config.NUMBER_OF_GENERATIONS_GCA}")
print(f"LGA Iterations: {config.NUMBER_OF_GENERATIONS_LGA}")
print("="*70 + "\n")

global_ga_function()

print("\n" + "="*70)
print("Checking T2_ga.json for correct message sizes...")
print("="*70 + "\n")

# Go back and check solution
os.chdir('..')
import json

with open('solution/T2_ga.json', 'r') as f:
    solution = json.load(f)

print("Dependencies in solution:")
for task in solution:
    if task['dependencies']:
        for dep in task['dependencies']:
            print(f"  Task {dep['task_id']} -> Task {task['task_id']}: message_size = {dep['message_size']}")

print("\nExpected: All message sizes should be 24.0 (or 0.0 for same-node)")
print("Bug would show: 25, 28, or other incorrect values")
