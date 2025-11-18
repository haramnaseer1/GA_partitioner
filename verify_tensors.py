"""Verify PyTorch tensor data for GNN training"""
import torch

data = torch.load('training_data.pt', weights_only=False)
print(f'Loaded {len(data)} graphs')
print(f'\nFirst graph statistics:')
print(f'  Nodes: {data[0]["num_nodes"]}')
print(f'  Edges: {data[0]["num_edges"]}')
print(f'  Node features shape: {data[0]["x"].shape}')
print(f'  Edge index shape: {data[0]["edge_index"].shape}')
print(f'  Edge attr shape: {data[0]["edge_attr"].shape}')
print(f'  Makespan: {data[0]["y"].item():.2f}')

print(f'\nFeature descriptions (6 per node):')
print(f'  [0] processing_time')
print(f'  [1] deadline')
print(f'  [2] start_time')
print(f'  [3] end_time')
print(f'  [4] processor_id')
print(f'  [5] is_scheduled (1.0 if scheduled, 0.0 otherwise)')

print(f'\nTotal statistics:')
print(f'  Total graphs: {len(data)}')
print(f'  Total nodes: {sum(d["num_nodes"] for d in data)}')
print(f'  Total edges: {sum(d["num_edges"] for d in data)}')
print(f'  Avg makespan: {sum(d["y"].item() for d in data)/len(data):.2f}')
print(f'\nReady for GNN training!')
