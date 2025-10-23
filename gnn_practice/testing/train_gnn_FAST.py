"""
FAST VERSION - GNN Training on LAMMPS Systems
Uses only 40 systems for quick testing (5 minutes instead of 20)
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

print("="*80)
print("FAST GNN TRAINING - QUICK TEST VERSION")
print("Using 40 systems (32 train / 8 test) for speed")
print("="*80)

# [Copy all the class definitions from the full script]
# SystemLoader, GraphBuilder, ImprovedGNN classes...
# [I'll include the optimized versions]

class SystemLoader:
    """Load and parse LAMMPS systems"""
    
    def __init__(self, data_dir="silica_systems_lammps"):
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        print(f"\nData directory: {self.data_dir.absolute()}")
    
    def load_all_systems(self, max_systems=None):
        """Load systems from directory"""
        
        system_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('system_')])
        
        if max_systems:
            system_dirs = system_dirs[:max_systems]
        
        print(f"\nLoading {len(system_dirs)} systems...")
        
        systems = []
        for i, system_dir in enumerate(system_dirs, 1):
            try:
                system_data = self.load_system(system_dir)
                systems.append(system_data)
                
                if i % 10 == 0:
                    print(f"  Loaded {i}/{len(system_dirs)} systems...")
            except Exception as e:
                print(f"  Warning: Failed to load {system_dir.name}: {e}")
        
        print(f"Successfully loaded {len(systems)} systems")
        
        return systems
    
    def load_system(self, system_dir):
        """Load one system"""
        
        props_file = system_dir / "properties.json"
        with open(props_file, 'r') as f:
            properties = json.load(f)
        
        dump_file = list(system_dir.glob("*.dump"))[0]
        atoms_data = self.parse_dump_file(dump_file)
        
        return {
            'system_id': properties['system_id'],
            'atoms': atoms_data,
            'youngs_modulus': properties['youngs_modulus_GPa'],
            'n_atoms': properties['n_atoms'],
            'density': properties['density_g_cm3'],
            'temperature': properties['temperature_K'],
            'box_size': properties['box_size_angstrom']
        }
    
    def parse_dump_file(self, dump_file):
        """Parse LAMMPS dump file"""
        
        with open(dump_file, 'r') as f:
            lines = f.readlines()
        
        atoms_line_idx = None
        for i, line in enumerate(lines):
            if 'ITEM: ATOMS' in line:
                atoms_line_idx = i
                break
        
        if atoms_line_idx is None:
            raise ValueError("Could not find ATOMS section")
        
        atoms = []
        for line in lines[atoms_line_idx + 1:]:
            parts = line.strip().split()
            if len(parts) >= 10:
                atoms.append({
                    'id': int(parts[0]),
                    'type': int(parts[1]),
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'z': float(parts[4]),
                    'vx': float(parts[5]),
                    'vy': float(parts[6]),
                    'vz': float(parts[7]),
                    'fx': float(parts[8]),
                    'fy': float(parts[9]),
                    'fz': float(parts[10])
                })
        
        return atoms


class GraphBuilder:
    """Build molecular graphs - OPTIMIZED VERSION"""
    
    def __init__(self, cutoff_distance=5.0, max_neighbors=12, max_atoms=800):
        self.cutoff_distance = cutoff_distance
        self.max_neighbors = max_neighbors
        self.max_atoms = max_atoms  # Limit graph size for speed
    
    def system_to_graph(self, system_data):
        """Convert system to graph"""
        
        atoms = system_data['atoms']
        
        # SPEED UP: Use only subset of atoms
        if len(atoms) > self.max_atoms:
            atoms = np.random.choice(atoms, self.max_atoms, replace=False).tolist()
        
        positions = np.array([[a['x'], a['y'], a['z']] for a in atoms])
        velocities = np.array([[a['vx'], a['vy'], a['vz']] for a in atoms])
        forces = np.array([[a['fx'], a['fy'], a['fz']] for a in atoms])
        atom_types = np.array([a['type'] for a in atoms])
        
        node_features = self._create_node_features(positions, velocities, forces, atom_types)
        edge_index = self._create_edges_fast(positions, system_data['box_size'])
        
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        y = torch.tensor([system_data['youngs_modulus']], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def _create_node_features(self, positions, velocities, forces, atom_types):
        """Create node features"""
        
        # Normalize
        pos_norm = (positions - positions.mean(axis=0)) / (positions.std(axis=0) + 1e-8)
        vel_norm = velocities / (velocities.std(axis=0) + 1e-8)
        force_norm = forces / (forces.std(axis=0) + 1e-8)
        
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        
        speed_norm = speeds / (speeds.std() + 1e-8)
        force_mag_norm = force_mags / (force_mags.std() + 1e-8)
        
        type_si = (atom_types == 1).astype(float).reshape(-1, 1)
        type_o = (atom_types == 2).astype(float).reshape(-1, 1)
        
        node_features = np.concatenate([
            pos_norm, vel_norm, force_norm,
            speed_norm, force_mag_norm,
            type_si, type_o
        ], axis=1)
        
        return node_features.astype(np.float32)
    
    def _create_edges_fast(self, positions, box_size):
        """Fast edge creation using vectorization"""
        
        n_atoms = len(positions)
        edges = []
        
        # Vectorized approach - much faster!
        for i in range(n_atoms):
            delta = positions - positions[i]
            delta = delta - box_size * np.round(delta / box_size)
            distances = np.linalg.norm(delta, axis=1)
            
            neighbor_mask = (distances < self.cutoff_distance) & (distances > 0)
            neighbor_indices = np.where(neighbor_mask)[0]
            
            if len(neighbor_indices) > self.max_neighbors:
                neighbor_distances = distances[neighbor_indices]
                sorted_idx = np.argsort(neighbor_distances)[:self.max_neighbors]
                neighbor_indices = neighbor_indices[sorted_idx]
            
            for j in neighbor_indices:
                edges.append([i, j])
        
        if len(edges) == 0:
            for i in range(min(30, n_atoms - 1)):
                edges.append([i, i + 1])
        
        return np.array(edges).T


class ImprovedGNN(nn.Module):
    """Improved GNN - smaller for faster training"""
    
    def __init__(self, input_dim, hidden_dim=64):
        super(ImprovedGNN, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 2)
        
        pooled_dim = hidden_dim * 2 * 3
        
        self.predictor = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.input_proj(x)
        x = self.bn_input(x)
        x = F.relu(x)
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        
        x = torch.cat([x_mean, x_max, x_add], dim=1)
        out = self.predictor(x)
        
        return out.squeeze(-1)


def extract_features(system_data):
    """Extract ML features"""
    atoms = system_data['atoms']
    positions = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    velocities = np.array([[a['vx'], a['vy'], a['vz']] for a in atoms])
    forces = np.array([[a['fx'], a['fy'], a['fz']] for a in atoms])
    atom_types = np.array([a['type'] for a in atoms])
    
    speeds = np.linalg.norm(velocities, axis=1)
    force_mags = np.linalg.norm(forces, axis=1)
    
    return {
        'n_atoms': len(atoms),
        'si_ratio': (atom_types == 1).mean(),
        'density': system_data['density'],
        'temperature': system_data['temperature'],
        'force_mean': force_mags.mean(),
        'force_std': force_mags.std(),
        'speed_mean': speeds.mean(),
        'speed_std': speeds.std(),
    }


def train_models(train_systems, test_systems, device):
    """Train all models"""
    
    # Traditional ML
    print("\n" + "="*80)
    print("TRAINING TRADITIONAL ML")
    print("="*80)
    
    X_train = np.array([list(extract_features(s).values()) for s in train_systems])
    y_train = np.array([s['youngs_modulus'] for s in train_systems])
    X_test = np.array([list(extract_features(s).values()) for s in test_systems])
    y_test = np.array([s['youngs_modulus'] for s in test_systems])
    
    # Ridge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    ml_results = {
        'Ridge': {'predictions': ridge_pred, 'r2': r2_score(y_test, ridge_pred), 'mae': mean_absolute_error(y_test, ridge_pred)},
        'Random Forest': {'predictions': rf_pred, 'r2': r2_score(y_test, rf_pred), 'mae': mean_absolute_error(y_test, rf_pred)}
    }
    
    print(f"Ridge: R² = {ml_results['Ridge']['r2']:.3f}, MAE = {ml_results['Ridge']['mae']:.2f}")
    print(f"Random Forest: R² = {ml_results['Random Forest']['r2']:.3f}, MAE = {ml_results['Random Forest']['mae']:.2f}")
    
    # GNN
    print("\n" + "="*80)
    print("TRAINING GNN")
    print("="*80)
    
    print("\nBuilding graphs (this may take 2-3 minutes)...")
    graph_builder = GraphBuilder(cutoff_distance=5.0, max_neighbors=12, max_atoms=800)
    
    train_graphs = [graph_builder.system_to_graph(s) for s in train_systems]
    test_graphs = [graph_builder.system_to_graph(s) for s in test_systems]
    
    print(f"Built {len(train_graphs)} training graphs, {len(test_graphs)} test graphs")
    
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)
    
    input_dim = train_graphs[0].x.shape[1]
    model = ImprovedGNN(input_dim=input_dim, hidden_dim=64).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"\nTraining GNN on {device}...")
    
    for epoch in range(100):  # Fewer epochs for speed
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            if pred.dim() > 1:
                pred = pred.squeeze()
            if batch.y.dim() > 1:
                batch.y = batch.y.squeeze()
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/100")
    
    # Test
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            predictions.extend(pred.cpu().numpy())
    
    predictions = np.array(predictions)
    gnn_r2 = r2_score(y_test, predictions)
    gnn_mae = mean_absolute_error(y_test, predictions)
    
    print(f"\nGNN: R² = {gnn_r2:.3f}, MAE = {gnn_mae:.2f}")
    
    ml_results['GNN'] = {'predictions': predictions, 'r2': gnn_r2, 'mae': gnn_mae}
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, result) in enumerate(ml_results.items()):
        ax = axes[idx]
        ax.scatter(y_test, result['predictions'], alpha=0.7, s=100)
        y_range = [y_test.min() - 2, y_test.max() + 2]
        ax.plot(y_range, y_range, 'r--', linewidth=2)
        ax.set_xlabel('True E (GPa)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted E (GPa)', fontsize=12, fontweight='bold')
        ax.set_title(f'{name}\nR² = {result["r2"]:.3f}, MAE = {result["mae"]:.2f} GPa', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('FAST_Results.png', dpi=200)
    print("\nSaved: FAST_Results.png")
    plt.close()
    
    return ml_results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load only 40 systems for speed
    loader = SystemLoader("silica_systems_lammps")
    all_systems = loader.load_all_systems(max_systems=40)
    
    # 32 train / 8 test
    np.random.seed(42)
    np.random.shuffle(all_systems)
    split_idx = 32
    
    train_systems = all_systems[:split_idx]
    test_systems = all_systems[split_idx:]
    
    print(f"\nTrain: {len(train_systems)}, Test: {len(test_systems)}")
    
    results = train_models(train_systems, test_systems, device)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nResults:")
    for name, res in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
        print(f"  {name:15s} | R² = {res['r2']:.3f} | MAE = {res['mae']:.2f} GPa")


if __name__ == "__main__":
    main()
