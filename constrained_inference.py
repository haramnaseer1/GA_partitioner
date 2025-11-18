"""
Constrained GNN Inference Module
=================================

Post-processes GNN predictions to enforce hard constraints:
1. Eligibility: Tasks only assigned to compatible processors
2. Precedence: Dependencies respected with communication delays
3. Duration: End = Start + processing_time/clock_speed
4. Non-overlap: Tasks serialized on same processor
5. Communication: Tier-based delays (Edge/Fog/Cloud)
"""

import torch
import json
import numpy as np
from pathlib import Path


class ConstrainedScheduler:
    """Enforces scheduling constraints on GNN predictions"""
    
    def __init__(self, platform_path='Path_Information/3_Tier_Platform.json'):
        """
        Initialize with platform information
        
        Args:
            platform_path: Path to platform JSON file
        """
        self.platform = self._load_platform(platform_path)
        self.processor_speeds = self._extract_processor_speeds()
        self.processor_types = self._extract_processor_types()
        
        # Resource type mapping (from GA code)
        self.resource_mapping = {
            1: 'General purpose CPU',
            2: 'FPGA',
            3: 'Raspberry Pi 5',
            4: 'Microcontroller',
            5: 'High Performance CPU',
            6: 'Graphical Processing Unit'
        }
        
        # Tier-based communication delays (from Guide.md Section 10.7)
        self.tier_delays = {
            ('Edge', 'Edge'): 50,
            ('Edge', 'Fog'): 150,
            ('Fog', 'Edge'): 150,
            ('Edge', 'Cloud'): 350,
            ('Cloud', 'Edge'): 350,
            ('Fog', 'Cloud'): 200,
            ('Cloud', 'Fog'): 200,
            ('Cloud', 'Cloud'): 175,
            ('Fog', 'Fog'): 150,  # Approximation
        }
    
    def _load_platform(self, platform_path):
        """Load platform configuration"""
        with open(platform_path, 'r') as f:
            return json.load(f)
    
    def _extract_processor_speeds(self):
        """Extract clock speeds for all processors"""
        speeds = {}
        for node in self.platform['platform']['nodes']:
            if not node.get('is_router', False):
                node_id = node['id']
                speed_str = node.get('clocking_speed', '1 GHz')
                speeds[node_id] = self._convert_clock_speed(speed_str)
        return speeds
    
    def _extract_processor_types(self):
        """Extract processor types for all nodes"""
        types = {}
        for node in self.platform['platform']['nodes']:
            if not node.get('is_router', False):
                types[node['id']] = node.get('type_of_processor', 'Unknown')
        return types
    
    def _convert_clock_speed(self, speed_str):
        """Convert clock speed string to Hz"""
        import re
        if 'GHz' in speed_str:
            match = re.findall(r'\d+\.?\d*', speed_str)
            return float(match[0]) * 1e9 if match else 1e9
        elif 'MHz' in speed_str:
            match = re.findall(r'\d+\.?\d*', speed_str)
            return float(match[0]) * 1e6 if match else 1e6
        else:
            return 1e9  # Default 1 GHz
    
    def _get_tier(self, node_id):
        """Get tier (Edge/Fog/Cloud) from node ID"""
        if 11 <= node_id <= 99:
            return 'Edge'
        elif 101 <= node_id <= 999:
            return 'Fog'
        elif node_id >= 1001:
            return 'Cloud'
        else:
            return 'Unknown'
    
    def _compute_comm_delay(self, sender_proc, receiver_proc, message_size):
        """Compute communication delay between two processors"""
        if sender_proc == receiver_proc:
            return 0.0  # Same node, no delay
        
        sender_tier = self._get_tier(sender_proc)
        receiver_tier = self._get_tier(receiver_proc)
        
        # Base tier delay
        tier_delay = self.tier_delays.get((sender_tier, receiver_tier), 100)
        
        # Total delay = message_size + tier_delay
        return float(message_size) + tier_delay
    
    def _create_eligibility_mask(self, can_run_on_list, num_processors=192):
        """
        Create binary mask for eligible processors
        
        Args:
            can_run_on_list: List of resource type IDs (e.g., [1, 2] for CPU, FPGA)
            num_processors: Total number of processors
        
        Returns:
            torch.Tensor: [num_processors] binary mask (1=eligible, 0=not)
        """
        mask = torch.zeros(num_processors)
        
        # Map resource types to actual processor IDs
        eligible_types = [self.resource_mapping[rt] for rt in can_run_on_list]
        
        for proc_id, proc_type in self.processor_types.items():
            if proc_type in eligible_types and proc_id < num_processors:
                mask[proc_id] = 1.0
        
        return mask
    
    def _topological_sort(self, messages):
        """
        Topological sort of tasks based on dependencies
        
        Args:
            messages: List of message dicts with 'sender' and 'receiver'
        
        Returns:
            List of task IDs in topological order
        """
        from collections import defaultdict, deque
        
        # Build adjacency list and in-degree count
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_tasks = set()
        
        for msg in messages:
            sender = msg['sender']
            receiver = msg['receiver']
            graph[sender].append(receiver)
            in_degree[receiver] += 1
            all_tasks.add(sender)
            all_tasks.add(receiver)
        
        # Initialize in-degree for all tasks
        for task in all_tasks:
            if task not in in_degree:
                in_degree[task] = 0
        
        # Kahn's algorithm
        queue = deque([task for task in all_tasks if in_degree[task] == 0])
        sorted_tasks = []
        
        while queue:
            task = queue.popleft()
            sorted_tasks.append(task)
            
            for neighbor in graph[task]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return sorted_tasks
    
    def apply_constraints(self, outputs, app_data, num_tasks):
        """
        Apply hard constraints to GNN predictions
        
        Args:
            outputs: Dict with 'processor', 'start_time', 'end_time', 'makespan'
            app_data: Application JSON data
            num_tasks: Number of tasks
        
        Returns:
            Dict with constrained predictions
        """
        # Clone outputs to avoid modifying originals
        constrained = {
            'processor': outputs['processor'].clone(),
            'start_time': outputs['start_time'].clone().squeeze(),
            'end_time': outputs['end_time'].clone().squeeze(),
            'makespan': outputs['makespan'].clone()
        }
        
        jobs = app_data['application']['jobs']
        messages = app_data['application']['messages']
        
        # ========================================
        # 1. ELIGIBILITY CONSTRAINT
        # ========================================
        for i, job in enumerate(jobs):
            if i >= num_tasks:
                break
            
            can_run_on = job.get('can_run_on', [1])  # Default to CPU
            mask = self._create_eligibility_mask(can_run_on, num_processors=192)
            
            # Mask invalid processors (set logits to -inf)
            constrained['processor'][i] = constrained['processor'][i] * mask + (-1e9) * (1 - mask)
        
        # Select processors (argmax after masking)
        processor_assignments = constrained['processor'].argmax(dim=1)
        
        # ========================================
        # 2. DURATION CONSISTENCY
        # ========================================
        durations = torch.zeros(num_tasks)
        for i, job in enumerate(jobs):
            if i >= num_tasks:
                break
            
            proc_id = processor_assignments[i].item()
            processing_time = job.get('processing_times', 100)  # Baseline at 1Hz
            clock_speed = self.processor_speeds.get(proc_id, 1e9)
            
            # Duration = processing_time / clock_speed
            durations[i] = processing_time / clock_speed
        
        # ========================================
        # 3. PRECEDENCE CONSTRAINT (Topological Order)
        # ========================================
        sorted_tasks = self._topological_sort(messages)
        
        # Build dependency map
        dependencies = {i: [] for i in range(num_tasks)}
        for msg in messages:
            sender = msg['sender']
            receiver = msg['receiver']
            if receiver < num_tasks and sender < num_tasks:
                dependencies[receiver].append({
                    'sender': sender,
                    'size': msg.get('size', 24)
                })
        
        # Enforce precedence in topological order
        for task_id in sorted_tasks:
            if task_id >= num_tasks:
                continue
            
            deps = dependencies.get(task_id, [])
            if deps:
                max_dep_finish = 0.0
                
                for dep in deps:
                    sender_id = dep['sender']
                    sender_proc = processor_assignments[sender_id].item()
                    receiver_proc = processor_assignments[task_id].item()
                    
                    # Communication delay
                    comm_delay = self._compute_comm_delay(
                        sender_proc, 
                        receiver_proc, 
                        dep['size']
                    )
                    
                    # Sender must finish + comm delay
                    dep_finish = constrained['end_time'][sender_id].item() + comm_delay
                    max_dep_finish = max(max_dep_finish, dep_finish)
                
                # Receiver cannot start before all dependencies finish
                constrained['start_time'][task_id] = max(
                    constrained['start_time'][task_id].item(),
                    max_dep_finish
                )
            
            # Update end time based on duration
            constrained['end_time'][task_id] = (
                constrained['start_time'][task_id] + durations[task_id]
            )
        
        # ========================================
        # 4. NON-OVERLAP CONSTRAINT (Per Processor)
        # ========================================
        for proc_id in range(192):
            # Find all tasks assigned to this processor
            tasks_on_proc = (processor_assignments == proc_id).nonzero(as_tuple=True)[0]
            
            if len(tasks_on_proc) <= 1:
                continue  # No overlap possible
            
            # Sort by start time
            sorted_indices = constrained['start_time'][tasks_on_proc].argsort()
            sorted_tasks_on_proc = tasks_on_proc[sorted_indices]
            
            # Serialize: each task starts after previous ends
            for i in range(1, len(sorted_tasks_on_proc)):
                prev_task = sorted_tasks_on_proc[i - 1]
                curr_task = sorted_tasks_on_proc[i]
                
                # Current task cannot start before previous ends
                min_start = constrained['end_time'][prev_task].item()
                if constrained['start_time'][curr_task].item() < min_start:
                    constrained['start_time'][curr_task] = min_start
                    constrained['end_time'][curr_task] = (
                        constrained['start_time'][curr_task] + durations[curr_task]
                    )
        
        # ========================================
        # 5. RECOMPUTE MAKESPAN
        # ========================================
        constrained['makespan'] = constrained['end_time'].max().unsqueeze(0)
        
        # Store processor assignments
        constrained['processor_assignments'] = processor_assignments
        
        return constrained
    
    def predict_with_constraints(self, model, data, app_path):
        """
        Run GNN prediction with constraint enforcement
        
        Args:
            model: Trained GNN model
            data: PyG Data object
            app_path: Path to application JSON
        
        Returns:
            Dict with constrained predictions
        """
        # Load application data
        with open(app_path, 'r') as f:
            app_data = json.load(f)
        
        num_tasks = len(app_data['application']['jobs'])
        
        # Get raw GNN predictions
        model.eval()
        with torch.no_grad():
            outputs = model(data)
        
        # Apply constraints
        constrained = self.apply_constraints(outputs, app_data, num_tasks)
        
        return constrained
    
    def to_solution_format(self, constrained, app_data):
        """
        Convert constrained predictions to GA solution format
        
        Args:
            constrained: Dict from apply_constraints()
            app_data: Application JSON data
        
        Returns:
            List of dicts in solution format
        """
        solution = []
        jobs = app_data['application']['jobs']
        messages = app_data['application']['messages']
        
        # Build dependency map with message info
        dep_map = {i: [] for i in range(len(jobs))}
        for msg in messages:
            receiver = msg['receiver']
            if receiver < len(jobs):
                dep_map[receiver].append({
                    'task_id': msg['sender'],
                    'path_id': '0',  # Placeholder (would need path lookup)
                    'message_size': float(msg.get('size', 24))
                })
        
        for i, job in enumerate(jobs):
            solution.append({
                'node_id': int(constrained['processor_assignments'][i].item()),
                'task_id': job['id'],
                'start_time': float(constrained['start_time'][i].item()),
                'end_time': float(constrained['end_time'][i].item()),
                'dependencies': dep_map.get(i, [])
            })
        
        # Sort by node_id, start_time, task_id
        solution.sort(key=lambda x: (x['node_id'], x['start_time'], x['task_id']))
        
        return solution


def test_constrained_inference():
    """Test the constrained inference on a sample"""
    print("="*70)
    print("TESTING CONSTRAINED GNN INFERENCE")
    print("="*70)
    
    # Load model and data
    from train_gnn_multitask import create_multitask_model
    model = create_multitask_model()
    
    # Try to load trained weights
    try:
        checkpoint = torch.load('models_multitask/best_model.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded trained model\n")
    except:
        print("⚠ Using untrained model (for structure testing)\n")
    
    # Load test data
    test_data = torch.load('training_data_multitask.pt')
    sample = test_data[0]
    
    # Initialize constrained scheduler
    scheduler = ConstrainedScheduler()
    
    # Run constrained prediction
    print("Running constrained prediction...")
    constrained = scheduler.apply_constraints(
        {
            'processor': sample.y_processor.unsqueeze(0) if sample.y_processor.dim() == 0 else model(sample)['processor'],
            'start_time': sample.y_start.unsqueeze(0) if hasattr(sample, 'y_start') else model(sample)['start_time'],
            'end_time': sample.y_end.unsqueeze(0) if hasattr(sample, 'y_end') else model(sample)['end_time'],
            'makespan': sample.y_makespan.unsqueeze(0) if hasattr(sample, 'y_makespan') else model(sample)['makespan']
        },
        {'application': {'jobs': [{'id': i, 'can_run_on': [1], 'processing_times': 100} for i in range(sample.x.size(0))],
                        'messages': []}},
        sample.x.size(0)
    )
    
    print(f"✓ Constrained prediction completed")
    print(f"  Processor assignments: {constrained['processor_assignments'].tolist()[:5]}...")
    print(f"  Start times: {constrained['start_time'][:5].tolist()}")
    print(f"  End times: {constrained['end_time'][:5].tolist()}")
    print(f"  Makespan: {constrained['makespan'].item():.2f}")
    print("\n✓ Constrained inference module working!")


if __name__ == '__main__':
    test_constrained_inference()
