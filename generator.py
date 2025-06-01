# This script generates a 4D tetrahedral mesh by extruding a 3D surface mesh from an OBJ file.
# It uses PyVista for mesh handling, TetGen for tetrahedralization, and NumPy for array operations.

# VERSION: 0.0.1 (not finished at all)
# AUTHOR: TechDye


# --------------- #
#     IMPORTS     #
# --------------- #

from pathlib import Path
import logging

import pyvista as pv
import tetgen
import numpy as np


# ------------- #
#     UTILS     #
# ------------- #

# Set up logging to console and file.
def _setup_logger(log_file):
	logger = logging.getLogger('Extrude4D')
	logger.setLevel(logging.DEBUG)
	
	# Clear any existing handlers to avoid duplicates
	logger.handlers.clear()
	
	# Console handler
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
	console_handler.setFormatter(console_formatter)
	logger.addHandler(console_handler)
	
	# File handler
	file_handler = logging.FileHandler(log_file)
	file_handler.setLevel(logging.DEBUG)
	file_formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
	file_handler.setFormatter(file_formatter)
	logger.addHandler(file_handler)
	
	return logger

# Load a 3D surface mesh from an OBJ file.
def _load_obj(file_path, logger):
	logger.info(f"Loading OBJ file: {file_path}")

	try:
		mesh = pv.read(file_path)
		logger.debug(f"Loaded mesh with {mesh.n_points} points and {mesh.n_cells} cells")

		return mesh
	except FileNotFoundError:
		logger.error(f"The file {file_path} does not exist")

		raise FileNotFoundError(f"The file {file_path} does not exist")
	except OSError as e:
		logger.error(f"Error reading the file {file_path}: {e}")
		
		raise OSError(f"Error reading the file {file_path}: {e}")


# ----------------- #
#     FUNCTIONS     #
# ----------------- #

# Tetrahedralize a 3D surface mesh.
def tetrahedralize_mesh(surface_mesh, logger):
	"""Tetrahedralize a 3D surface mesh using TetGen."""
	logger.info("Starting tetrahedralization")
	try:
		# Triangulate the mesh to ensure all faces are triangles
		logger.debug("Triangulating mesh")
		triangulated_mesh = surface_mesh.triangulate()
		
		# Perform tetrahedralization
		logger.debug("Initializing TetGen")
		tet = tetgen.TetGen(triangulated_mesh)
		nodes, elements = tet.tetrahedralize(
			order=1,		  # Linear tetrahedra
			mindihedral=20,   # Minimum dihedral angle for quality
			minratio=1.5,	 # Minimum radius-edge ratio
			quality=True	  # Enable mesh quality improvement
		)
		logger.info(f"Tetrahedralization complete: {nodes.shape[0]} nodes, {elements.shape[0]} tetrahedrons")
		logger.debug(f"Nodes shape: {nodes.shape}, Elements shape: {elements.shape}")
		return nodes, elements
	except Exception as e:
		logger.error(f"Tetrahedralization failed: {e}")
		raise

# Extrude a 3D tetrahedral mesh into 4D by duplicating along the w-axis.
def extrude_to_4d(nodes: np.ndarray, elements: np.ndarray, logger: logging.Logger, extrusion_distance=1.0, w_coord_func=lambda x, y, z: 0.0):
	logger.info(f"Starting 4D extrusion with distance {extrusion_distance}")
	n = nodes.shape[0]
	
	# Extend original nodes to 4D with w-coordinate
	logger.debug("Computing w-coordinates for nodes")
	w_coords = np.array([w_coord_func(x, y, z) for x, y, z in nodes])
	nodes_4d_base = np.column_stack((nodes, w_coords))  # At w=base
	nodes_4d_extruded = np.column_stack((nodes, w_coords + extrusion_distance))  # At w=base+d
	
	# Combine vertices
	nodes_4d = np.vstack((nodes_4d_base, nodes_4d_extruded))
	logger.debug(f"Created {nodes_4d.shape[0]} 4D nodes")
	
	# Create tetrahedrons
	elements_4d = []
	
	# 1) Original tetrahedrons at w=base
	elements_4d.extend(elements.tolist())
	logger.debug(f"Added {len(elements)} original tetrahedrons")
	
	# 2) Extruded tetrahedrons at w=base+d
	elements_top = elements + n  # Offset indices for extruded vertices
	elements_4d.extend(elements_top.tolist())
	logger.debug(f"Added {len(elements)} extruded tetrahedrons")
	
	# 3) Connecting tetrahedrons for each face
	logger.debug("Extracting faces from tetrahedrons")
	faces = []
	for tet in elements:
		faces.extend([
			sorted([tet[0], tet[1], tet[2]]),
			sorted([tet[0], tet[1], tet[3]]),
			sorted([tet[0], tet[2], tet[3]]),
			sorted([tet[1], tet[2], tet[3]])
		])
	
	# Remove duplicates
	faces = np.unique(faces, axis=0)
	logger.debug(f"Extracted {len(faces)} unique faces")
	
	# Create connecting tetrahedrons for each face
	logger.debug("Creating connecting tetrahedrons")
	for face in faces:
		i, j, k = face
		elements_4d.append([i, j, k, i + n])
		elements_4d.append([i, j, k, j + n])
		elements_4d.append([i, j, k, k + n])
	
	elements_4d = np.array(elements_4d)
	logger.info(f"4D extrusion complete: {elements_4d.shape[0]} tetrahedrons")
	logger.debug(f"4D elements shape: {elements_4d.shape}")
	
	return nodes_4d, elements_4d

# Export the 4D coordinates of the 4 vertices for each tetrahedron
def export_tetrahedrons_4d(nodes_4d, elements_4d, output_file, logger):
	logger.info(f"Exporting 4D tetrahedrons to {output_file}")
	try:
		with open(output_file, 'w') as f:
			for i, tetra in enumerate(elements_4d, start=1):
				f.write(f"{i}:\n")
				for vertex_idx in tetra:
					x, y, z, w = nodes_4d[vertex_idx]
					f.write(f"{x:.6f} {y:.6f} {z:.6f} {w:.6f}\n")
				f.write("\n")
		logger.info(f"Export complete: {len(elements_4d)} tetrahedrons written")
	except Exception as e:
		logger.error(f"Failed to export tetrahedrons: {e}")
		raise

def generate_tetrahedrons_4d(input_file, output_file, extrusion_distance=1.0, w_coord_func=lambda x, y, z: 0.0):
	logger = _setup_logger(Path("extrude_tetrahedrons_4d.log"))
	logger.info("Starting 4D tetrahedral mesh generation")
	
	surface_mesh = _load_obj(input_file, logger)
	nodes_3d, elements_3d = tetrahedralize_mesh(surface_mesh, logger)
	nodes_4d, elements_4d = extrude_to_4d(nodes_3d, elements_3d, logger, extrusion_distance, w_coord_func)
	export_tetrahedrons_4d(nodes_4d, elements_4d, output_file, logger)
	
	logger.info("4D tetrahedral mesh generation completed successfully.")


# ------------ #
#     MAIN     #
# ------------ #

def main():
	CUR_FOLDER = Path(__file__).parent

	input_path = CUR_FOLDER / input("Enter the path to your input OBJ file: ")
	if not input_path.exists():
		print(f"ERROR: The file {input_path} does not exist.")
		return
	elif input_path.suffix.lower() != ".obj":
		print(f"ERROR: The file {input_path} is not an OBJ file.")
		return
	
	output_path = CUR_FOLDER / input("Enter the desired output file path: ")
	if output_path.is_dir():
		print("ERROR: The output path is a directory, please provide a file path.")
		return
	if output_path.exists():
		print(f"WARNING: The file {output_path} exist.")

		y = input("Do you want to overwrite it? (y/n): ").strip().lower()
		if y.lower() != 'y':
			print("Exiting without overwriting.")
			return

	extrusion_distance_str: str = ""

	while not extrusion_distance_str.replace(".", "").isnumeric():
		extrusion_distance_str = input("Enter the extrusion distance (default is 1.0): ")
		if extrusion_distance_str == "": break
	
	if extrusion_distance_str: extrusion_distance: float = float(extrusion_distance_str)
	else: extrusion_distance: float = 1.0

	generate_tetrahedrons_4d(
		input_file=input_path,  
		output_file=output_path,  
		extrusion_distance=extrusion_distance, 
		w_coord_func=lambda x, y, z: 0.0 
	)

if __name__ == "__main__":
	main()