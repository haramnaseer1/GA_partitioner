"""
GNN Data Preprocessing Script
==============================

This script preprocesses the GA scheduling data for GNN training by:
1. Normalizing clock speeds and processing times
2. Mapping resource IDs to categorical indices
3. Encoding platform tiers (Edge/Fog/Cloud)
4. Creating normalized feature vectors

Input:
    - Application/*.json (task DAGs)
    - Platform/EdgeAI-Trust_Platform.json (platform topology)
    - solution/*_ga.json (GA scheduling solutions)

Output:
    - gnn_solution/preprocessed_data.pkl (normalized training data)
    - gnn_solution/feature_stats.json (normalization statistics)

Usage:
    python Script/preprocess_gnn_data.py
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

# Resource type mapping
RESOURCE_TYPES = {
    'General purpose CPU': 1,
    'FPGA': 2,
    'Raspberry Pi 5': 3,
    'Microcontroller': 4,
    'High Performance CPU': 5,
    'GPU': 6
}

# Tier classification based on node ID ranges
def get_tier(node_id: int) -> int:
    """
    Classify node into tier based on ID range.
    Returns: 0=Edge, 1=Fog, 2=Cloud
    """
    if 11 <= node_id <= 99:
        return 0  # Edge
    elif 101 <= node_id <= 999:
        return 1  # Fog
    elif node_id >= 1001:
        return 2  # Cloud
    else:
        return 0  # Default to Edge


def parse_clock_speed(speed_str: str) -> float:
    """
    Parse clock speed string to MHz value.
    Examples: "333 MHz" -> 333.0, "2.5 GHz" -> 2500.0
    """
    speed_str = speed_str.strip()
    if 'GHz' in speed_str:
        return float(speed_str.replace('GHz', '').strip()) * 1000
    elif 'MHz' in speed_str:
        return float(speed_str.replace('MHz', '').strip())
    else:
        return float(speed_str)


def load_platform(platform_path: str) -> Dict:
    """Load and parse platform model."""
    with open(platform_path, 'r') as f:
        platform = json.load(f)
    
    # Parse platform information
    nodes_info = {}
    for node in platform['platform']['nodes']:
        node_id = node['id']
        nodes_info[node_id] = {
            'processor_type': node['type_of_processor'],
            'clock_speed_mhz': parse_clock_speed(node['clocking_speed']),
            'tier': get_tier(node_id),
            'is_router': node.get('is_router', False)
        }
    
    return nodes_info


def load_application(app_path: str) -> Dict:
    """Load and parse application model."""
    with open(app_path, 'r') as f:
        app_data = json.load(f)
    
    jobs = {}
    for job in app_data['application']['jobs']:
        jobs[job['id']] = {
            'processing_time': job.get('processing_times', job.get('wcet_fullspeed', 0)),
            'deadline': job.get('deadline', 0),
            'can_run_on': job.get('can_run_on', [])
        }
    
    messages = {}
    for msg in app_data['application'].get('messages', []):
        messages[msg['id']] = {
            'sender': msg['sender'],
            'receiver': msg['receiver'],
            'size': msg['size']
        }
    
    return {'jobs': jobs, 'messages': messages}


def load_solution(solution_path: str) -> List[Dict]:
    """Load GA scheduling solution."""
    with open(solution_path, 'r') as f:
        solution = json.load(f)
    return solution


def preprocess_dataset(app_dir: str, platform_path: str, solution_dir: str) -> Dict:
    """
    Preprocess entire dataset.
    
    Returns:
        Dict containing:
            - samples: List of preprocessed training samples
            - feature_stats: Normalization statistics
            - metadata: Dataset metadata
    """
    # Load platform
    platform_nodes = load_platform(platform_path)
    
    # Collect all features for normalization
    all_processing_times = []
    all_clock_speeds = [info['clock_speed_mhz'] for info in platform_nodes.values()]
    all_start_times = []
    all_end_times = []
    all_message_sizes = []
    
    samples = []
    
    # Get all application files
    app_files = sorted([f for f in os.listdir(app_dir) if f.endswith('.json')])
    
    print(f"Processing {len(app_files)} applications...")
    
    for idx, app_file in enumerate(app_files, 1):
        app_name = app_file.replace('.json', '')
        solution_file = f"{app_name}_ga.json"
        
        app_path = os.path.join(app_dir, app_file)
        solution_path = os.path.join(solution_dir, solution_file)
        
        # Skip if solution doesn't exist
        if not os.path.exists(solution_path):
            print(f"  [{idx}/{len(app_files)}] SKIP {app_file} - No solution")
            continue
        
        try:
            # Load data
            app_data = load_application(app_path)
            solution_data = load_solution(solution_path)
            
            # Create sample
            sample = {
                'app_name': app_name,
                'num_jobs': len(app_data['jobs']),
                'num_messages': len(app_data['messages']),
                'jobs': app_data['jobs'],
                'messages': app_data['messages'],
                'solution': solution_data,
                'platform_nodes': platform_nodes
            }
            
            samples.append(sample)
            
            # Collect statistics
            for job in app_data['jobs'].values():
                all_processing_times.append(job['processing_time'])
            
            for task in solution_data:
                all_start_times.append(task['start_time'])
                all_end_times.append(task['end_time'])
                for dep in task.get('dependencies', []):
                    all_message_sizes.append(dep['message_size'])
            
            if idx % 10 == 0:
                print(f"  [{idx}/{len(app_files)}] Processed {app_file}")
        
        except Exception as e:
            print(f"  [{idx}/{len(app_files)}] ERROR {app_file}: {str(e)}")
            continue
    
    # Calculate normalization statistics
    feature_stats = {
        'processing_time': {
            'min': float(np.min(all_processing_times)) if all_processing_times else 0,
            'max': float(np.max(all_processing_times)) if all_processing_times else 1,
            'mean': float(np.mean(all_processing_times)) if all_processing_times else 0,
            'std': float(np.std(all_processing_times)) if all_processing_times else 1
        },
        'clock_speed': {
            'min': float(np.min(all_clock_speeds)) if all_clock_speeds else 0,
            'max': float(np.max(all_clock_speeds)) if all_clock_speeds else 1,
            'mean': float(np.mean(all_clock_speeds)) if all_clock_speeds else 0,
            'std': float(np.std(all_clock_speeds)) if all_clock_speeds else 1
        },
        'start_time': {
            'min': float(np.min(all_start_times)) if all_start_times else 0,
            'max': float(np.max(all_start_times)) if all_start_times else 1,
            'mean': float(np.mean(all_start_times)) if all_start_times else 0,
            'std': float(np.std(all_start_times)) if all_start_times else 1
        },
        'end_time': {
            'min': float(np.min(all_end_times)) if all_end_times else 0,
            'max': float(np.max(all_end_times)) if all_end_times else 1,
            'mean': float(np.mean(all_end_times)) if all_end_times else 0,
            'std': float(np.std(all_end_times)) if all_end_times else 1
        },
        'message_size': {
            'min': float(np.min(all_message_sizes)) if all_message_sizes else 0,
            'max': float(np.max(all_message_sizes)) if all_message_sizes else 1,
            'mean': float(np.mean(all_message_sizes)) if all_message_sizes else 0,
            'std': float(np.std(all_message_sizes)) if all_message_sizes else 1
        }
    }
    
    metadata = {
        'num_samples': len(samples),
        'num_resource_types': len(RESOURCE_TYPES),
        'num_tiers': 3,  # Edge, Fog, Cloud
        'total_applications': len(app_files),
        'resource_type_mapping': RESOURCE_TYPES
    }
    
    return {
        'samples': samples,
        'feature_stats': feature_stats,
        'metadata': metadata
    }


def normalize_features(samples: List[Dict], feature_stats: Dict) -> List[Dict]:
    """
    Apply min-max normalization to features.
    
    Normalization formula: (x - min) / (max - min)
    """
    print("\nNormalizing features...")
    
    for sample in samples:
        # Normalize job processing times
        for job_id, job in sample['jobs'].items():
            pt_min = feature_stats['processing_time']['min']
            pt_max = feature_stats['processing_time']['max']
            if pt_max > pt_min:
                job['processing_time_norm'] = (job['processing_time'] - pt_min) / (pt_max - pt_min)
            else:
                job['processing_time_norm'] = 0.0
        
        # Normalize platform node clock speeds
        for node_id, node in sample['platform_nodes'].items():
            cs_min = feature_stats['clock_speed']['min']
            cs_max = feature_stats['clock_speed']['max']
            if cs_max > cs_min:
                node['clock_speed_norm'] = (node['clock_speed_mhz'] - cs_min) / (cs_max - cs_min)
            else:
                node['clock_speed_norm'] = 0.0
            
            # Encode resource type as one-hot
            node['resource_type_id'] = RESOURCE_TYPES.get(node['processor_type'], 0)
            
            # Tier is already encoded as 0/1/2
        
        # Normalize solution times
        for task in sample['solution']:
            st_min = feature_stats['start_time']['min']
            st_max = feature_stats['start_time']['max']
            et_min = feature_stats['end_time']['min']
            et_max = feature_stats['end_time']['max']
            
            if st_max > st_min:
                task['start_time_norm'] = (task['start_time'] - st_min) / (st_max - st_min)
            else:
                task['start_time_norm'] = 0.0
            
            if et_max > et_min:
                task['end_time_norm'] = (task['end_time'] - et_min) / (et_max - et_min)
            else:
                task['end_time_norm'] = 0.0
        
        # Normalize message sizes
        for msg_id, msg in sample['messages'].items():
            ms_min = feature_stats['message_size']['min']
            ms_max = feature_stats['message_size']['max']
            if ms_max > ms_min:
                msg['size_norm'] = (msg['size'] - ms_min) / (ms_max - ms_min)
            else:
                msg['size_norm'] = 0.0
    
    print(f"✅ Normalized {len(samples)} samples")
    return samples


def main():
    """Main preprocessing pipeline."""
    print("="*80)
    print("GNN Data Preprocessing")
    print("="*80)
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    app_dir = os.path.join(repo_root, 'Application')
    platform_path = os.path.join(repo_root, 'Platform', 'EdgeAI-Trust_Platform.json')
    solution_dir = os.path.join(repo_root, 'solution')
    output_dir = os.path.join(repo_root, 'gnn_solution')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Preprocess dataset
    print("\nStep 1: Loading and collecting statistics...")
    dataset = preprocess_dataset(app_dir, platform_path, solution_dir)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {dataset['metadata']['num_samples']}")
    print(f"  Resource types: {dataset['metadata']['num_resource_types']}")
    print(f"  Platform tiers: {dataset['metadata']['num_tiers']}")
    
    # Normalize features
    print("\nStep 2: Normalizing features...")
    dataset['samples'] = normalize_features(dataset['samples'], dataset['feature_stats'])
    
    # Save preprocessed data
    output_pkl = os.path.join(output_dir, 'preprocessed_data.pkl')
    output_stats = os.path.join(output_dir, 'feature_stats.json')
    
    print(f"\nStep 3: Saving preprocessed data...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"  ✅ Saved: {output_pkl}")
    
    with open(output_stats, 'w') as f:
        json.dump({
            'feature_stats': dataset['feature_stats'],
            'metadata': dataset['metadata']
        }, f, indent=2)
    print(f"  ✅ Saved: {output_stats}")
    
    print("\n" + "="*80)
    print("✅ Preprocessing Complete!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {output_pkl}")
    print(f"  - {output_stats}")
    print(f"\nReady for GNN training!")


if __name__ == "__main__":
    main()
