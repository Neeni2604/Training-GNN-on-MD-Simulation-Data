import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import json
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_silica_structure(n_atoms=50000, box_size=100.0):
    """
    Generate a toy amorphous silica structure with approximately n_atoms atoms.
    Silica has SiO2 composition, so we'll have ~16667 Si atoms and ~33333 O atoms.
    """
    # Calculate number of Si and O atoms (SiO2 stoichiometry)
    n_si = int(n_atoms / 3)  # ~16667 Si atoms
    n_o = n_atoms - n_si     # ~33333 O atoms
    
    # Generate random positions in the simulation box
    positions = np.random.uniform(0, box_size, (n_atoms, 3))
    
    # Create atom types (1 for Si, 2 for O)
    atom_types = np.concatenate([np.ones(n_si, dtype=int), 
                                np.ones(n_o, dtype=int) * 2])
    
    # Create atom IDs
    atom_ids = np.arange(1, n_atoms + 1)
    
    return positions, atom_types, atom_ids, n_si, n_o

def calculate_distances_batch(positions, batch_size=5000):
    """Calculate distances in batches to handle memory for large systems"""
    n_atoms = len(positions)
    distances = np.zeros((n_atoms, n_atoms))
    
    for i in range(0, n_atoms, batch_size):
        end_i = min(i + batch_size, n_atoms)
        for j in range(0, n_atoms, batch_size):
            end_j = min(j + batch_size, n_atoms)
            
            # Calculate distances for this batch
            pos_i = positions[i:end_i]
            pos_j = positions[j:end_j]
            
            # Vectorized distance calculation
            diff = pos_i[:, np.newaxis, :] - pos_j[np.newaxis, :, :]
            batch_dist = np.sqrt(np.sum(diff**2, axis=2))
            
            distances[i:end_i, j:end_j] = batch_dist
    
    return distances

def generate_forces_and_energies_optimized(positions, atom_types, box_size):
    """
    Generate realistic forces and energies for large silica system (optimized)
    """
    n_atoms = len(positions)
    forces = np.zeros((n_atoms, 3))
    
    # Lennard-Jones parameters for Si-O interactions (simplified)
    epsilon_si_o = 0.005  # eV
    sigma_si_o = 3.0      # Angstroms
    epsilon_o_o = 0.003   # eV
    sigma_o_o = 2.8       # Angstroms
    
    potential_energy = 0.0
    cutoff = min(8.0, box_size/4)  # Reasonable cutoff
    
    print(f"Calculating forces for {n_atoms} atoms with cutoff {cutoff:.1f} Å...")
    
    # Process in chunks to manage memory
    chunk_size = 1000
    for i in range(0, n_atoms, chunk_size):
        end_i = min(i + chunk_size, n_atoms)
        
        # Calculate distances from current chunk to all atoms
        pos_i = positions[i:end_i]
        
        # Vectorized calculation for this chunk
        diff = pos_i[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        # Apply cutoff mask
        mask = (distances > 0) & (distances < cutoff)
        
        for local_i, global_i in enumerate(range(i, end_i)):
            neighbors = np.where(mask[local_i])[0]
            
            for j in neighbors:
                if j <= global_i:  # Avoid double counting
                    continue
                    
                r = distances[local_i, j]
                
                # Choose parameters based on atom types
                if atom_types[global_i] != atom_types[j]:  # Si-O interaction
                    eps, sig = epsilon_si_o, sigma_si_o
                else:  # Same type interaction
                    eps, sig = epsilon_o_o, sigma_o_o
                
                # Lennard-Jones force calculation
                r6 = (sig/r)**6
                r12 = r6**2
                
                # Force magnitude
                f_mag = 24 * eps * (2*r12 - r6) / r
                
                # Force direction (unit vector)
                dr = positions[j] - positions[global_i]
                dr_unit = dr / r
                
                # Apply Newton's third law
                force = f_mag * dr_unit
                forces[global_i] += force
                forces[j] -= force
                
                # Add to potential energy
                potential_energy += 4 * eps * (r12 - r6)
        
        if i % (chunk_size * 5) == 0:
            print(f"Processed {min(end_i, n_atoms)}/{n_atoms} atoms...")
    
    # Add random thermal noise (scaled for larger system)
    forces += np.random.normal(0, 0.0005, forces.shape)
    
    return forces, potential_energy

def generate_velocities(n_atoms, atom_types, temperature=300.0):
    """Generate Maxwell-Boltzmann distributed velocities"""
    # Boltzmann constant in eV/K
    kb = 8.617e-5
    
    # Masses in atomic units (Si: 28.0855, O: 15.9994)
    masses = np.array([28.0855 if atom_type == 1 else 15.9994 
                      for atom_type in atom_types])
    
    # Generate velocities from Maxwell-Boltzmann distribution
    velocities = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        sigma_v = np.sqrt(kb * temperature / masses[i])
        velocities[i] = np.random.normal(0, sigma_v, 3)
    
    return velocities

def generate_stress_strain_data(n_steps=50):
    """Generate stress-strain data for mechanical property extraction"""
    # Generate strain values (small deformations)
    strain = np.linspace(0, 0.05, n_steps)
    
    # Typical values for silica glass
    youngs_modulus = 70.0  # GPa
    poissons_ratio = 0.17
    
    # Generate stress (with some noise to make it realistic)
    stress = youngs_modulus * strain + np.random.normal(0, 0.5, n_steps)
    
    # Generate lateral strain (for Poisson's ratio calculation)
    lateral_strain = -poissons_ratio * strain + np.random.normal(0, 0.001, n_steps)
    
    return strain, stress, lateral_strain

# Generate the main MD dataset with 50,000 atoms
n_atoms = 50000
box_size = 100.0  # Larger box to accommodate more atoms
print(f"=== Generating Enhanced Toy Silica MD Dataset with {n_atoms} atoms ===")

positions, atom_types, atom_ids, n_si, n_o = generate_silica_structure(n_atoms, box_size)
print(f"Generated {n_si} Si atoms and {n_o} O atoms")

# Generate forces and energies (optimized for large system)
forces, potential_energy = generate_forces_and_energies_optimized(positions, atom_types, box_size)
print("Force calculation completed")

# Generate velocities
print("Generating velocities...")
velocities = generate_velocities(n_atoms, atom_types)

# Create the main dataframe
print("Creating dataframe...")
md_data = pd.DataFrame({
    'atom_id': atom_ids,
    'atom_type': atom_types,  # 1 = Si, 2 = O
    'x': positions[:, 0],
    'y': positions[:, 1],
    'z': positions[:, 2],
    'vx': velocities[:, 0],
    'vy': velocities[:, 1],
    'vz': velocities[:, 2],
    'fx': forces[:, 0],
    'fy': forces[:, 1],
    'fz': forces[:, 2],
})

# Add atom type labels
md_data['element'] = md_data['atom_type'].map({1: 'Si', 2: 'O'})

# Generate stress-strain data
strain, stress, lateral_strain = generate_stress_strain_data()
mechanical_data = pd.DataFrame({
    'strain': strain,
    'stress_GPa': stress,
    'lateral_strain': lateral_strain
})

# Calculate some derived properties
print("Calculating derived properties...")
md_data['speed'] = np.sqrt(md_data['vx']**2 + md_data['vy']**2 + md_data['vz']**2)
md_data['force_magnitude'] = np.sqrt(md_data['fx']**2 + md_data['fy']**2 + md_data['fz']**2)

# System properties
system_properties = {
    'total_atoms': n_atoms,
    'n_silicon': n_si,
    'n_oxygen': n_o,
    'box_size': box_size,
    'temperature': 300.0,
    'potential_energy': potential_energy,
    'kinetic_energy': 0.5 * np.sum(md_data['speed']**2),
    'density': n_atoms / (box_size**3),
    'composition': 'SiO2',
    'structure_type': 'amorphous'
}

print("\n=== Enhanced Toy Silica MD Dataset Statistics ===")
print(f"Total atoms: {n_atoms}")
print(f"Silicon atoms: {n_si}")
print(f"Oxygen atoms: {n_o}")
print(f"Box size: {box_size} Å")
print(f"Potential energy: {potential_energy:.3f} eV")
print(f"System density: {system_properties['density']:.6f} atoms/Å³")
print()

print("=== Atomic Data Sample ===")
print(md_data.head(10))
print()

print("=== System Properties ===")
for key, value in system_properties.items():
    print(f"{key}: {value}")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save data to files in the same directory as the script
md_data_path = os.path.join(script_dir, 'silica_md_atoms.csv')
mechanical_data_path = os.path.join(script_dir, 'silica_mechanical_data.csv')
system_properties_path = os.path.join(script_dir, 'system_properties.json')

print(f"\n=== Saving Files ===")
print("Saving atomic data (this may take a moment for 50,000 atoms)...")
md_data.to_csv(md_data_path, index=False)

print("Saving mechanical data...")
mechanical_data.to_csv(mechanical_data_path, index=False)

print("Saving system properties...")
with open(system_properties_path, 'w') as f:
    json.dump(system_properties, f, indent=2)

print("\n=== Files Generated ===")
print(f"1. {md_data_path} - Atomic positions, velocities, forces (50,000 atoms)")
print(f"2. {mechanical_data_path} - Stress-strain data for ML training")
print(f"3. {system_properties_path} - System metadata")

print("\n=== Data Description ===")
print("Columns in silica_md_atoms.csv:")
print("- atom_id: Unique atom identifier (1-50000)")
print("- atom_type: 1=Silicon, 2=Oxygen")
print("- x,y,z: Atomic positions (Å) in 100×100×100 box")
print("- vx,vy,vz: Atomic velocities (Å/fs)")
print("- fx,fy,fz: Forces on atoms (eV/Å)")
print("- element: Element symbol (Si/O)")
print("- speed: Velocity magnitude")
print("- force_magnitude: Force magnitude")
print()
print("This large dataset should provide sufficient data for:")
print("- Training robust GNN models (>10,000 training samples possible)")
print("- Creating diverse molecular environments")
print("- Learning complex structure-property relationships")
print("- Achieving good ML performance on mechanical property prediction")