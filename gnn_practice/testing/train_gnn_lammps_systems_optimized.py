"""
GNN for Predicting Young's Modulus from LAMMPS Systems
Trains on 100 silica systems, predicts Young's modulus for new systems
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

print("="*80)
print("GNN FOR YOUNG'S MODULUS PREDICTION FROM LAMMPS SYSTEMS")
print("="*80)

class SystemLoader:
    """Load and parse LAMMPS systems"""
    
    def __init__(self, data_dir="silica_systems_lammps"):
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        print(f"\nData directory: {self.data_dir.absolute()}")
    
    def load_all_systems(self):
        """Load all systems from directory"""
        
        systems = []
        
        # Get all system directories
        system_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('system_')])
        
        print(f"\nFound {len(system_dirs)} systems")
        print("Loading systems...")
        
        for i, system_dir in enumerate(system_dirs, 1):
            try:
                system_data = self.load_system(system_dir)
                systems.append(system_data)
                
                if i % 20 == 0:
                    print(f"  Loaded {i}/{len(system_dirs)} systems...")
            except Exception as e:
                print(f"  Warning: Failed to load {system_dir.name}: {e}")
        
        print(f"\nSuccessfully loaded {len(systems)} systems")
        
        return systems
    
    def load_system(self, system_dir):
        """Load one system from directory"""
        
        # Load properties (contains Young's modulus target)
        props_file = system_dir / "properties.json"
        with open(props_file, 'r') as f:
            properties = json.load(f)
        
        # Load dump file (contains positions, velocities, forces)
        dump_file = list(system_dir.glob("*.dump"))[0]
        atoms_data = self.parse_dump_file(dump_file)
        
        system_data = {
            'system_id': properties['system_id'],
            'atoms': atoms_data,
            'youngs_modulus': properties['youngs_modulus_GPa'],
            'n_atoms': properties['n_atoms'],
            'density': properties['density_g_cm3'],
            'temperature': properties['temperature_K'],
            'box_size': properties['box_size_angstrom']
        }
        
        return system_data
    
    def parse_dump_file(self, dump_file):
        """Parse LAMMPS dump file to extract atom data"""
        
        with open(dump_file, 'r') as f:
            lines = f.readlines()
        
        # Find where atom data starts
        atoms_line_idx = None
        for i, line in enumerate(lines):
            if 'ITEM: ATOMS' in line:
                atoms_line_idx = i
                break
        
        if atoms_line_idx is None:
            raise ValueError("Could not find ATOMS section in dump file")
        
        # Parse atom data
        atoms = []
        for line in lines[atoms_line_idx + 1:]:
            parts = line.strip().split()
            if len(parts) >= 10:  # id type x y z vx vy vz fx fy fz
                atom = {
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
                }
                atoms.append(atom)
        
        return atoms


class GraphBuilder:
    """Build molecular graphs from atomic systems"""
    
    def __init__(self, cutoff_distance=5.0, max_neighbors=15):
        self.cutoff_distance = cutoff_distance
        self.max_neighbors = max_neighbors
    
    def system_to_graph(self, system_data):
        """Convert system to PyTorch Geometric graph"""
        
        atoms = system_data['atoms']
        n_atoms = len(atoms)
        box_size = system_data['box_size']
        
        # Extract positions, velocities, forces
        positions = np.array([[a['x'], a['y'], a['z']] for a in atoms])
        velocities = np.array([[a['vx'], a['vy'], a['vz']] for a in atoms])
        forces = np.array([[a['fx'], a['fy'], a['fz']] for a in atoms])
        atom_types = np.array([a['type'] for a in atoms])
        
        # Create node features
        node_features = self._create_node_features(
            positions, velocities, forces, atom_types
        )
        
        # Create edges based on distance
        edge_index = self._create_edges(positions, box_size)
        
        # Create PyG Data object
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        y = torch.tensor([system_data['youngs_modulus']], dtype=torch.float)
        
        graph = Data(x=x, edge_index=edge_index, y=y)
        
        return graph
    
    def _create_node_features(self, positions, velocities, forces, atom_types):
        """Create feature vector for each atom"""
        
        n_atoms = len(positions)
        
        # Normalize features
        pos_mean = positions.mean(axis=0)
        pos_std = positions.std(axis=0) + 1e-8
        positions_norm = (positions - pos_mean) / pos_std
        
        vel_std = velocities.std(axis=0) + 1e-8
        velocities_norm = velocities / vel_std
        
        force_std = forces.std(axis=0) + 1e-8
        forces_norm = forces / force_std
        
        # Calculate derived features
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        
        speed_norm = speeds / (speeds.std() + 1e-8)
        force_mag_norm = force_mags / (force_mags.std() + 1e-8)
        
        # One-hot encode atom types (1=Si, 2=O)
        type_si = (atom_types == 1).astype(float).reshape(-1, 1)
        type_o = (atom_types == 2).astype(float).reshape(-1, 1)
        
        # Concatenate all features
        node_features = np.concatenate([
            positions_norm,      # 3 features
            velocities_norm,     # 3 features
            forces_norm,         # 3 features
            speed_norm,          # 1 feature
            force_mag_norm,      # 1 feature
            type_si,             # 1 feature
            type_o               # 1 feature
        ], axis=1)
        
        return node_features.astype(np.float32)
    
    def _create_edges(self, positions, box_size):
        """Create edges based on distance cutoff - OPTIMIZED VERSION"""
        
        n_atoms = len(positions)
        
        # CRITICAL: Limit to max 2500 atoms for graph construction
        if n_atoms > 2500:
            sample_indices = np.random.choice(n_atoms, 2500, replace=False)
            positions = positions[sample_indices]
            n_atoms = 2500
        
        edges = []
        
        # Vectorized distance calculation (much faster!)
        for i in range(n_atoms):
            # Calculate all distances from atom i
            delta = positions - positions[i]
            
            # Periodic boundary conditions
            delta = delta - box_size * np.round(delta / box_size)
            distances = np.linalg.norm(delta, axis=1)
            
            # Find neighbors within cutoff (excluding self)
            neighbor_mask = (distances < self.cutoff_distance) & (distances > 0)
            neighbor_indices = np.where(neighbor_mask)[0]
            
            # Sort by distance and keep closest
            if len(neighbor_indices) > self.max_neighbors:
                neighbor_distances = distances[neighbor_indices]
                sorted_idx = np.argsort(neighbor_distances)[:self.max_neighbors]
                neighbor_indices = neighbor_indices[sorted_idx]
            
            # Add edges
            for j in neighbor_indices:
                edges.append([i, j])
        
        # Ensure at least some edges exist
        if len(edges) == 0:
            for i in range(min(50, n_atoms - 1)):
                edges.append([i, i + 1])
                edges.append([i + 1, i])
        
        edge_index = np.array(edges).T
        
        return edge_index


class ImprovedGNN(nn.Module):
    """Improved GNN for predicting Young's modulus from atomic systems"""
    
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(ImprovedGNN, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        
        # Graph convolution layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 2)
        
        self.conv4 = GCNConv(hidden_dim * 2, hidden_dim * 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim * 2)
        
        # Residual connection
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        
        # Global pooling combines mean, max, and sum
        pooled_dim = hidden_dim * 2 * 3
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 3),
            
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        x = self.bn_input(x)
        x = F.relu(x)
        x_input = x
        
        # First conv layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second conv layer with residual
        x_residual = self.residual_proj(x_input)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x + x_residual)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Third conv layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Fourth conv layer
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Global pooling (3 types)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        
        # Concatenate pooled features
        x = torch.cat([x_mean, x_max, x_add], dim=1)
        
        # Predict Young's modulus
        out = self.predictor(x)
        
        return out.squeeze(-1)


def extract_traditional_ml_features(system_data):
    """Extract engineered features for traditional ML models"""
    
    atoms = system_data['atoms']
    
    # Extract arrays
    positions = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    velocities = np.array([[a['vx'], a['vy'], a['vz']] for a in atoms])
    forces = np.array([[a['fx'], a['fy'], a['fz']] for a in atoms])
    atom_types = np.array([a['type'] for a in atoms])
    
    # Calculate features
    n_si = np.sum(atom_types == 1)
    n_o = np.sum(atom_types == 2)
    si_ratio = n_si / len(atom_types)
    
    speeds = np.linalg.norm(velocities, axis=1)
    force_mags = np.linalg.norm(forces, axis=1)
    
    features = {
        'n_atoms': len(atoms),
        'si_ratio': si_ratio,
        'density': system_data['density'],
        'temperature': system_data['temperature'],
        'box_size': system_data['box_size'],
        
        'force_mean': force_mags.mean(),
        'force_std': force_mags.std(),
        'force_max': force_mags.max(),
        
        'speed_mean': speeds.mean(),
        'speed_std': speeds.std(),
        
        'vel_x_mean': velocities[:, 0].mean(),
        'vel_y_mean': velocities[:, 1].mean(),
        'vel_z_mean': velocities[:, 2].mean(),
    }
    
    return features


def train_traditional_ml(train_systems, test_systems):
    """Train traditional ML models for comparison"""
    
    print("\n" + "="*80)
    print("TRAINING TRADITIONAL ML MODELS")
    print("="*80)
    
    # Extract features
    print("\nExtracting features...")
    X_train = np.array([list(extract_traditional_ml_features(s).values()) for s in train_systems])
    y_train = np.array([s['youngs_modulus'] for s in train_systems])
    
    X_test = np.array([list(extract_traditional_ml_features(s).values()) for s in test_systems])
    y_test = np.array([s['youngs_modulus'] for s in test_systems])
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Define models
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Scale features for Ridge
        if 'Ridge' in name:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'predictions': y_pred,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"  MAE:  {mae:.2f} GPa")
        print(f"  RMSE: {rmse:.2f} GPa")
        print(f"  R²:   {r2:.3f}")
    
    return results, y_test


def train_gnn(train_systems, test_systems, device):
    """Train GNN model"""
    
    print("\n" + "="*80)
    print("TRAINING GNN MODEL")
    print("="*80)
    
    # Build graphs
    print("\nBuilding molecular graphs...")
    graph_builder = GraphBuilder(cutoff_distance=5.0, max_neighbors=15)
    
    train_graphs = []
    print(f"Converting {len(train_systems)} training systems to graphs...")
    for i, system in enumerate(train_systems, 1):
        graph = graph_builder.system_to_graph(system)
        train_graphs.append(graph)
        if i % 10 == 0:
            print(f"  Built {i}/{len(train_systems)} training graphs...")
    
    test_graphs = []
    print(f"\nConverting {len(test_systems)} test systems to graphs...")
    for i, system in enumerate(test_systems, 1):
        graph = graph_builder.system_to_graph(system)
        test_graphs.append(graph)
        if i % 5 == 0:
            print(f"  Built {i}/{len(test_systems)} test graphs...")
    
    print(f"\nBuilt {len(train_graphs)} training graphs")
    print(f"Built {len(test_graphs)} test graphs")
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)
    
    # Initialize model
    input_dim = train_graphs[0].x.shape[1]
    model = ImprovedGNN(input_dim=input_dim, hidden_dim=128, dropout=0.3).to(device)
    
    print(f"\nModel architecture:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: 128")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining on {device}...")
    print("="*80)
    
    num_epochs = 200
    best_loss = float('inf')
    patience_counter = 0
    patience = 35
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        
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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Best = {best_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING GNN ON TEST SET")
    print("="*80)
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)
            if batch.y.dim() > 1:
                batch.y = batch.y.squeeze()
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Clip predictions to realistic range
    predictions = np.clip(predictions, 50, 90)
    
    # Metrics
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    print(f"\nGNN Test Results:")
    print(f"  MAE:  {mae:.2f} GPa")
    print(f"  RMSE: {rmse:.2f} GPa")
    print(f"  R²:   {r2:.3f}")
    
    return {
        'predictions': predictions,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }, targets


def plot_results(ml_results, gnn_results, y_test):
    """Plot prediction results for all models"""
    
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    # Combine all results
    all_results = {**ml_results, 'Improved GNN': gnn_results}
    
    # Create individual plots
    for model_name, result in all_results.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        
        preds = result['predictions']
        
        # Scatter plot
        ax.scatter(y_test, preds, alpha=0.6, s=100, edgecolors='black', linewidth=1, c='#4FC3E0')
        
        # Perfect prediction line
        y_range = [min(y_test.min(), preds.min()) - 2, 
                   max(y_test.max(), preds.max()) + 2]
        ax.plot(y_range, y_range, 'r--', linewidth=3, label='Perfect Prediction')
        
        # Labels
        ax.set_xlabel('True Young\'s Modulus (GPa)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Predicted Young\'s Modulus (GPa)', fontsize=16, fontweight='bold')
        
        title_color = 'darkgreen' if 'GNN' in model_name else 'black'
        ax.set_title(f'{model_name}\nR² = {result["r2"]:.3f}, MAE = {result["mae"]:.2f} GPa, RMSE = {result["rmse"]:.2f} GPa\n' +
                    f'(Tested on {len(y_test)} systems)',
                    fontsize=14, fontweight='bold', color=title_color)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='upper left')
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save
        safe_name = model_name.replace(' ', '_')
        plt.savefig(f'{safe_name}_results.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {safe_name}_results.png")
        
        plt.close()
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, (model_name, result) in enumerate(all_results.items()):
        ax = axes[idx]
        preds = result['predictions']
        
        ax.scatter(y_test, preds, alpha=0.6, s=80, edgecolors='black', linewidth=0.8, c='#4FC3E0')
        
        y_range = [min(y_test.min(), preds.min()) - 2, 
                   max(y_test.max(), preds.max()) + 2]
        ax.plot(y_range, y_range, 'r--', linewidth=2.5, label='Perfect')
        
        ax.set_xlabel('True E (GPa)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted E (GPa)', fontsize=14, fontweight='bold')
        
        title_color = 'darkgreen' if 'GNN' in model_name else 'black'
        ax.set_title(f'{model_name}\nR² = {result["r2"]:.3f}, MAE = {result["mae"]:.2f} GPa',
                    fontsize=13, fontweight='bold', color=title_color)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig('All_Models_Comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: All_Models_Comparison.png")
    plt.close()
    
    print("\nAll plots saved successfully!")


def main():
    """Main training and evaluation pipeline"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load systems
    print("\n" + "="*80)
    print("LOADING LAMMPS SYSTEMS")
    print("="*80)
    
    loader = SystemLoader("silica_systems_lammps")
    all_systems = loader.load_all_systems()
    
    print(f"\nDataset summary:")
    print(f"  Total systems: {len(all_systems)}")
    print(f"  Young's modulus range: {min(s['youngs_modulus'] for s in all_systems):.1f} - {max(s['youngs_modulus'] for s in all_systems):.1f} GPa")
    print(f"  Average atoms per system: {np.mean([s['n_atoms'] for s in all_systems]):.0f}")
    
    # 80/20 train/test split
    np.random.seed(42)
    np.random.shuffle(all_systems)
    
    split_idx = int(0.8 * len(all_systems))
    train_systems = all_systems[:split_idx]
    test_systems = all_systems[split_idx:]
    
    print(f"\n" + "="*80)
    print("TRAIN/TEST SPLIT")
    print("="*80)
    print(f"  Training systems: {len(train_systems)}")
    print(f"  Test systems: {len(test_systems)}")
    print(f"  Split ratio: 80/20")
    
    # Train traditional ML
    ml_results, y_test = train_traditional_ml(train_systems, test_systems)
    
    # Train GNN
    gnn_results, y_test_gnn = train_gnn(train_systems, test_systems, device)
    
    # Print final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    
    all_results = {**ml_results, 'Improved GNN': gnn_results}
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    print(f"\nRanking by R² score:")
    for rank, (name, result) in enumerate(sorted_results, 1):
        marker = "🏆" if rank == 1 else "⭐" if 'GNN' in name else "  "
        print(f"{rank}. {marker} {name:20s} | R² = {result['r2']:.3f} | MAE = {result['mae']:.2f} GPa | RMSE = {result['rmse']:.2f} GPa")
    
    # Plot results
    plot_results(ml_results, gnn_results, y_test)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nTrained on {len(train_systems)} systems, tested on {len(test_systems)} systems")
    print(f"Best model: {sorted_results[0][0]} (R² = {sorted_results[0][1]['r2']:.3f})")
    print("\nGenerated plots:")
    print("  - Ridge_Regression_results.png")
    print("  - Random_Forest_results.png")
    print("  - Gradient_Boosting_results.png")
    print("  - Improved_GNN_results.png")
    print("  - All_Models_Comparison.png")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()