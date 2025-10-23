The files outside the 'testing' folder can be ignored. They are wrong, but might contain useful information. The main code is in the testing folder.

My most recent GNN code is in the file called 'train_gnn_lammps_systems_optimized.py'. It is in the testing folder. The GNN needs to be improved upon. Right now, it is performing worse than other traditional ML techniques. It needs to perform better to support my hypothesis.

The 'train_gnn_FAST.py' is a GNN that runs faster, but it is not learning optimally.

The 'generate_lammps_systems.py' generates synthetic MD Simulation data of Silica systems. It creates 100 silica systems that can be found in the 'silica_systems_lammps' folder (which is in the 'testing' folder)

Here is the order I run the code files in:
- generate_lammps_systems.py
- validate_and_visualize_dataset.py (visualizes the silica systems)
- train_gnn_lammps_systems_optimized.py
