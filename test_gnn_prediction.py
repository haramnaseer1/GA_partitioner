"""
Test the trained GNN model on a single application
Compare GNN prediction vs actual GA solution
"""

import torch
import json
import sys
from pathlib import Path
from train_gnn_scheduling import create_model
from create_tensors import load_application, create_node_features, create_edge_index

def load_trained_model(model_path='models_scheduling/best_model.pt'):
    """Load the trained GNN model"""
    model = create_model()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_makespan(model, app_path):
    """Predict makespan for an application using trained GNN"""
    # Load application
    app_data = load_application(app_path)
    
    # Create empty schedule for features (GNN predicts before scheduling)
    schedule = {}
    
    # Create graph features
    node_features = create_node_features(app_data, schedule)
    edge_index, edge_attr = create_edge_index(app_data)
    
    # Already tensors from create_edge_index
    x = node_features
    edge_index_tensor = edge_index
    edge_attr_tensor = edge_attr
    
    # Create batch
    from torch_geometric.data import Data
    data = Data(
        x=x,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr_tensor,
        batch=torch.zeros(x.size(0), dtype=torch.long)
    )
    
    # Predict
    with torch.no_grad():
        prediction = model(data)
    
    return prediction.item()

def get_actual_makespan(solution_path):
    """Get actual makespan from GA solution"""
    with open(solution_path, 'r') as f:
        solution = json.load(f)
    return max(task['end_time'] for task in solution)

def test_prediction(app_name='T2_var_001'):
    """Test GNN prediction on a specific application"""
    print("="*70)
    print(f"Testing GNN on: {app_name}")
    print("="*70)
    print()
    
    # Paths
    app_path = f'Application/{app_name}.json'
    solution_path = f'solution/{app_name}_ga.json'
    
    # Check if files exist
    if not Path(app_path).exists():
        print(f"❌ Application file not found: {app_path}")
        return
    
    if not Path(solution_path).exists():
        print(f"⚠ Solution file not found: {solution_path}")
        print("Generating GA solution first...")
        import subprocess
        subprocess.run(['python', '-m', 'src.main', '0', app_path], 
                      capture_output=True, timeout=120)
        subprocess.run(['python', 'src/simplify.py', '--input', app_path, 
                       '--log', 'Logs/global_ga.log'],
                      capture_output=True, timeout=60)
        
        if not Path(solution_path).exists():
            print(f"❌ Failed to generate solution")
            return
        print("✓ Solution generated\n")
    
    # Load model
    print("Loading trained GNN model...")
    model = load_trained_model()
    print("✓ Model loaded\n")
    
    # Get GNN prediction
    print("Running GNN prediction...")
    predicted_makespan = predict_makespan(model, app_path)
    print(f"✓ GNN Predicted Makespan: {predicted_makespan:.2f}\n")
    
    # Get actual makespan
    print("Reading actual GA solution...")
    actual_makespan = get_actual_makespan(solution_path)
    print(f"✓ Actual GA Makespan: {actual_makespan:.2f}\n")
    
    # Calculate error
    error = abs(predicted_makespan - actual_makespan)
    error_pct = (error / actual_makespan) * 100
    
    print("="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"GNN Prediction:    {predicted_makespan:>10.2f}")
    print(f"Actual Makespan:   {actual_makespan:>10.2f}")
    print(f"Absolute Error:    {error:>10.2f}")
    print(f"Percentage Error:  {error_pct:>10.2f}%")
    print()
    
    # Verdict
    if error_pct < 5:
        print("✓✓✓ EXCELLENT - Error < 5%")
    elif error_pct < 10:
        print("✓✓ VERY GOOD - Error < 10%")
    elif error_pct < 20:
        print("✓ GOOD - Error < 20%")
    else:
        print("○ MODERATE - Error > 20%")
    
    print("="*70)
    print()
    
    return {
        'app_name': app_name,
        'predicted': predicted_makespan,
        'actual': actual_makespan,
        'error': error,
        'error_pct': error_pct
    }

if __name__ == '__main__':
    # Test on specified app or default
    app_name = sys.argv[1] if len(sys.argv) > 1 else 'T2_var_001'
    test_prediction(app_name)
