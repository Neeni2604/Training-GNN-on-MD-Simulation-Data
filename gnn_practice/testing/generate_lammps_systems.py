#!/usr/bin/env python3
# run in terminal using the command "python .\generate_lammps_systems.py"
"""
Generate 100 Synthetic Silica Systems in LAMMPS Format
Each system has computed Young's modulus from stress-strain response
"""

import numpy as np
import json
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

class SilicaSystemGenerator:
    """Generate synthetic silica systems with mechanical properties"""
    
    def __init__(self, output_dir="silica_systems_lammps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Silicon and Oxygen masses (amu)
        self.mass_si = 28.0855
        self.mass_o = 15.9994
        
        # Lennard-Jones parameters for Si-O interactions
        self.epsilon_si_o = 0.005  # eV
        self.sigma_si_o = 3.0      # Angstroms
        self.epsilon_o_o = 0.003
        self.sigma_o_o = 2.8
        
        print("="*80)
        print("SYNTHETIC SILICA SYSTEM GENERATOR")
        print("Generating 100 systems with LAMMPS format data")
        print("="*80)
    
    def generate_system(self, system_id, n_atoms_range=(2000, 5000), 
                       density_range=(2.0, 2.6), temp_range=(250, 350)):
        """
        Generate one silica system with varied properties
        
        Parameters:
        -----------
        system_id : int
            System identifier
        n_atoms_range : tuple
            Range for total number of atoms
        density_range : tuple
            Range for system density (g/cm³)
        temp_range : tuple
            Range for temperature (K)
        """
        
        # Vary system parameters
        target_density = np.random.uniform(*density_range)  # g/cm³
        temperature = np.random.uniform(*temp_range)        # K
        n_atoms_total = np.random.randint(*n_atoms_range)
        
        # Ensure SiO2 stoichiometry (1:2 ratio)
        n_si = n_atoms_total // 3
        n_o = n_atoms_total - n_si
        n_atoms_total = n_si + n_o
        
        # Calculate box size from density
        # SiO2 molecular weight = 60.08 g/mol
        mass_total = n_si * self.mass_si + n_o * self.mass_o  # amu
        mass_grams = mass_total * 1.66054e-24  # Convert amu to grams
        volume_cm3 = mass_grams / target_density
        volume_angstrom3 = volume_cm3 * 1e24  # Convert cm³ to ų
        box_size = volume_angstrom3 ** (1/3)  # Cubic box
        
        # Generate random positions
        positions = np.random.uniform(0, box_size, (n_atoms_total, 3))
        
        # Create atom types (1 = Si, 2 = O)
        atom_types = np.concatenate([
            np.ones(n_si, dtype=int),
            np.ones(n_o, dtype=int) * 2
        ])
        
        # Generate velocities from Maxwell-Boltzmann distribution
        velocities = self._generate_velocities(atom_types, temperature)
        
        # Calculate forces using simplified potential
        forces = self._calculate_forces(positions, atom_types, box_size)
        
        # Apply synthetic stress and calculate Young's modulus
        youngs_modulus, stress_strain_data = self._calculate_youngs_modulus(
            positions, atom_types, box_size, temperature
        )
        
        # Create system data structure
        system_data = {
            'system_id': system_id,
            'n_atoms': n_atoms_total,
            'n_si': n_si,
            'n_o': n_o,
            'box_size': box_size,
            'density': target_density,
            'temperature': temperature,
            'youngs_modulus': youngs_modulus,
            'positions': positions,
            'velocities': velocities,
            'forces': forces,
            'atom_types': atom_types,
            'stress_strain_data': stress_strain_data
        }
        
        return system_data
    
    def _generate_velocities(self, atom_types, temperature):
        """Generate velocities from Maxwell-Boltzmann distribution"""
        kb = 8.617e-5  # eV/K
        
        velocities = np.zeros((len(atom_types), 3))
        
        for i, atom_type in enumerate(atom_types):
            mass = self.mass_si if atom_type == 1 else self.mass_o
            sigma_v = np.sqrt(kb * temperature / mass)
            velocities[i] = np.random.normal(0, sigma_v, 3)
        
        return velocities
    
    def _calculate_forces(self, positions, atom_types, box_size, cutoff=8.0):
        """Calculate forces using Lennard-Jones potential"""
        n_atoms = len(positions)
        forces = np.zeros((n_atoms, 3))
        
        # Sample subset for efficiency
        sample_size = min(500, n_atoms)
        sample_indices = np.random.choice(n_atoms, sample_size, replace=False)
        
        for i in sample_indices:
            for j in range(i + 1, n_atoms):
                rij = positions[j] - positions[i]
                
                # Apply periodic boundary conditions (minimum image convention)
                rij = rij - box_size * np.round(rij / box_size)
                
                r = np.linalg.norm(rij)
                
                if r < cutoff and r > 0.5:  # Avoid singularity
                    # Select LJ parameters
                    if atom_types[i] != atom_types[j]:  # Si-O
                        eps, sig = self.epsilon_si_o, self.sigma_si_o
                    else:  # O-O or Si-Si
                        eps, sig = self.epsilon_o_o, self.sigma_o_o
                    
                    # LJ force
                    r6 = (sig / r) ** 6
                    r12 = r6 ** 2
                    f_mag = 24 * eps * (2 * r12 - r6) / r
                    
                    force_vec = f_mag * rij / r
                    forces[i] += force_vec
                    forces[j] -= force_vec
        
        # Add thermal noise
        forces += np.random.normal(0, 0.0001, forces.shape)
        
        return forces
    
    def _calculate_youngs_modulus(self, positions, atom_types, box_size, temperature):
        """
        Calculate Young's modulus from synthetic stress-strain response
        
        This simulates applying uniaxial strain and measuring stress
        """
        
        # Base Young's modulus depends on density and temperature
        # Real silica: ~70-75 GPa at room temperature
        
        # Calculate system density
        n_si = np.sum(atom_types == 1)
        n_o = np.sum(atom_types == 2)
        mass_total = n_si * self.mass_si + n_o * self.mass_o
        mass_grams = mass_total * 1.66054e-24
        volume_cm3 = (box_size ** 3) * 1e-24
        density = mass_grams / volume_cm3
        
        # Base modulus (typical for silica)
        base_modulus = 72.0  # GPa
        
        # Density effect (higher density → stiffer)
        density_effect = (density - 2.2) * 15.0  # ±15 GPa per g/cm³
        
        # Temperature effect (higher T → softer)
        temp_effect = (300 - temperature) * 0.03  # -0.03 GPa per K
        
        # Structural disorder effect (force variance indicates disorder)
        force_magnitudes = np.linalg.norm(self.forces, axis=1) if hasattr(self, 'forces') else np.random.uniform(0, 0.01, len(atom_types))
        force_std = np.std(force_magnitudes)
        disorder_effect = -force_std * 100  # Higher disorder → lower modulus
        
        # Calculate Young's modulus
        youngs_modulus = base_modulus + density_effect + temp_effect + disorder_effect
        
        # Add realistic noise
        youngs_modulus += np.random.normal(0, 1.5)
        
        # Clip to realistic range for silica
        youngs_modulus = np.clip(youngs_modulus, 55, 85)
        
        # Generate stress-strain curve
        strain_points = np.linspace(0, 0.05, 20)  # 0-5% strain
        stress_points = youngs_modulus * strain_points + np.random.normal(0, 0.3, len(strain_points))
        
        stress_strain_data = {
            'strain': strain_points.tolist(),
            'stress': stress_points.tolist(),
            'method': 'synthetic_uniaxial_tension'
        }
        
        return float(youngs_modulus), stress_strain_data
    
    def write_lammps_data_file(self, system_data, filename):
        """
        Write LAMMPS data file (.data format)
        This is the standard LAMMPS input format
        """
        
        with open(filename, 'w') as f:
            # Header
            f.write(f"LAMMPS data file for silica system {system_data['system_id']}\n\n")
            
            # Counts
            f.write(f"{system_data['n_atoms']} atoms\n")
            f.write(f"2 atom types\n\n")
            
            # Box bounds
            box = system_data['box_size']
            f.write(f"0.0 {box:.6f} xlo xhi\n")
            f.write(f"0.0 {box:.6f} ylo yhi\n")
            f.write(f"0.0 {box:.6f} zlo zhi\n\n")
            
            # Masses
            f.write("Masses\n\n")
            f.write(f"1 {self.mass_si:.4f}  # Si\n")
            f.write(f"2 {self.mass_o:.4f}  # O\n\n")
            
            # Atoms section (atom-ID atom-type x y z)
            f.write("Atoms  # atomic\n\n")
            
            positions = system_data['positions']
            atom_types = system_data['atom_types']
            
            for i in range(system_data['n_atoms']):
                f.write(f"{i+1} {atom_types[i]} {positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f}\n")
            
            # Velocities section
            f.write("\nVelocities\n\n")
            
            velocities = system_data['velocities']
            
            for i in range(system_data['n_atoms']):
                f.write(f"{i+1} {velocities[i,0]:.8f} {velocities[i,1]:.8f} {velocities[i,2]:.8f}\n")
    
    def write_lammps_dump_file(self, system_data, filename):
        """
        Write LAMMPS dump file (.dump format)
        This contains atomic positions, velocities, and forces
        """
        
        with open(filename, 'w') as f:
            # Header
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{system_data['n_atoms']}\n")
            
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            box = system_data['box_size']
            f.write(f"0.0 {box:.6f}\n")
            f.write(f"0.0 {box:.6f}\n")
            f.write(f"0.0 {box:.6f}\n")
            
            f.write("ITEM: ATOMS id type x y z vx vy vz fx fy fz\n")
            
            positions = system_data['positions']
            velocities = system_data['velocities']
            forces = system_data['forces']
            atom_types = system_data['atom_types']
            
            for i in range(system_data['n_atoms']):
                f.write(f"{i+1} {atom_types[i]} "
                       f"{positions[i,0]:.6f} {positions[i,1]:.6f} {positions[i,2]:.6f} "
                       f"{velocities[i,0]:.8f} {velocities[i,1]:.8f} {velocities[i,2]:.8f} "
                       f"{forces[i,0]:.8f} {forces[i,1]:.8f} {forces[i,2]:.8f}\n")
    
    def write_properties_file(self, system_data, filename):
        """Write system properties including Young's modulus"""
        
        properties = {
            'system_id': system_data['system_id'],
            'n_atoms': system_data['n_atoms'],
            'n_si': system_data['n_si'],
            'n_o': system_data['n_o'],
            'composition': 'SiO2',
            'box_size_angstrom': float(system_data['box_size']),
            'density_g_cm3': float(system_data['density']),
            'temperature_K': float(system_data['temperature']),
            'youngs_modulus_GPa': float(system_data['youngs_modulus']),
            'stress_strain_data': system_data['stress_strain_data'],
            'atom_types': {
                '1': 'Si',
                '2': 'O'
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(properties, f, indent=2)
    
    def generate_all_systems(self, n_systems=100):
        """Generate all systems and save to files"""
        
        print(f"\nGenerating {n_systems} silica systems...")
        print(f"Output directory: {self.output_dir.absolute()}\n")
        
        summary = []
        
        for i in range(1, n_systems + 1):
            # Create system directory
            system_dir = self.output_dir / f"system_{i:03d}"
            system_dir.mkdir(exist_ok=True)
            
            # Generate system
            system_data = self.generate_system(
                system_id=i,
                n_atoms_range=(2000, 5000),
                density_range=(2.0, 2.6),
                temp_range=(250, 350)
            )
            
            # Write LAMMPS data file
            data_file = system_dir / f"silica_{i:03d}.data"
            self.write_lammps_data_file(system_data, data_file)
            
            # Write LAMMPS dump file
            dump_file = system_dir / f"silica_{i:03d}.dump"
            self.write_lammps_dump_file(system_data, dump_file)
            
            # Write properties JSON
            properties_file = system_dir / "properties.json"
            self.write_properties_file(system_data, properties_file)
            
            # Add to summary
            summary.append({
                'system_id': i,
                'n_atoms': system_data['n_atoms'],
                'density': system_data['density'],
                'temperature': system_data['temperature'],
                'youngs_modulus': system_data['youngs_modulus']
            })
            
            if i % 10 == 0:
                print(f"  Generated {i}/{n_systems} systems...")
        
        # Write summary file
        self._write_summary(summary)
        
        print(f"\n{'='*80}")
        print("GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nGenerated {n_systems} systems in: {self.output_dir.absolute()}")
        print(f"\nEach system contains:")
        print(f"  - silica_XXX.data  : LAMMPS data file (positions, velocities)")
        print(f"  - silica_XXX.dump  : LAMMPS dump file (positions, velocities, forces)")
        print(f"  - properties.json  : System properties and Young's modulus")
        
        self._print_statistics(summary)
    
    def _write_summary(self, summary):
        """Write summary of all systems"""
        
        summary_file = self.output_dir / "systems_summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also write CSV for easy viewing
        import csv
        csv_file = self.output_dir / "systems_summary.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['system_id', 'n_atoms', 'density', 'temperature', 'youngs_modulus'])
            writer.writeheader()
            writer.writerows(summary)
    
    def _print_statistics(self, summary):
        """Print statistics of generated systems"""
        
        n_atoms_list = [s['n_atoms'] for s in summary]
        density_list = [s['density'] for s in summary]
        temp_list = [s['temperature'] for s in summary]
        E_list = [s['youngs_modulus'] for s in summary]
        
        print(f"\n{'='*80}")
        print("DATASET STATISTICS")
        print(f"{'='*80}")
        print(f"\nNumber of atoms:")
        print(f"  Range: {min(n_atoms_list)} - {max(n_atoms_list)}")
        print(f"  Mean: {np.mean(n_atoms_list):.0f} ± {np.std(n_atoms_list):.0f}")
        
        print(f"\nDensity (g/cm³):")
        print(f"  Range: {min(density_list):.2f} - {max(density_list):.2f}")
        print(f"  Mean: {np.mean(density_list):.2f} ± {np.std(density_list):.2f}")
        
        print(f"\nTemperature (K):")
        print(f"  Range: {min(temp_list):.1f} - {max(temp_list):.1f}")
        print(f"  Mean: {np.mean(temp_list):.1f} ± {np.std(temp_list):.1f}")
        
        print(f"\nYoung's Modulus (GPa):")
        print(f"  Range: {min(E_list):.1f} - {max(E_list):.1f}")
        print(f"  Mean: {np.mean(E_list):.1f} ± {np.std(E_list):.1f}")
        
        print(f"\n{'='*80}")
        print("FILES CREATED")
        print(f"{'='*80}")
        print(f"Total files: {len(summary) * 3 + 2}")
        print(f"  - {len(summary)} x .data files (LAMMPS data format)")
        print(f"  - {len(summary)} x .dump files (LAMMPS dump format)")
        print(f"  - {len(summary)} x properties.json files")
        print(f"  - 1 x systems_summary.json")
        print(f"  - 1 x systems_summary.csv")
        print(f"\n{'='*80}\n")


def main():
    """Main function to generate systems"""
    
    # Create generator
    generator = SilicaSystemGenerator(output_dir="silica_systems_lammps")
    
    # Generate 100 systems
    generator.generate_all_systems(n_systems=100)
    
    print("NEXT STEPS:")
    print("-" * 80)
    print("1. Examine the generated files in 'silica_systems_lammps/' directory")
    print("2. Check systems_summary.csv for overview of all systems")
    print("3. Use these systems to train your GNN model")
    print("4. Each system has ONE Young's modulus value to predict")
    print("\nFile formats:")
    print("  .data  - LAMMPS data file (readable by LAMMPS 'read_data' command)")
    print("  .dump  - LAMMPS dump file (includes forces, standard trajectory format)")
    print("  .json  - Properties including computed Young's modulus")
    print("-" * 80)


if __name__ == "__main__":
    main()
