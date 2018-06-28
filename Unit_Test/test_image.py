import pandas as pd
import numpy as np
from rdkit import Chem
import image

def test_cs_compute_features():
	mol = Chem.MolFromSmiles("CCN(CC)CCCl")

	# check mol
	if mol is None:
		raise Exception("mol is Illegal!")

	# check output
	mol, df_atom, df_bond, nancheckflag = image.cs_compute_features(mol)
	if df_atom.empty:
		raise Exception("Function is broken! No atom in the dataframe.")
	if df_bond.empty:
		raise Exception("Function is broken! No bond in the dataframe.")


def test_cs_set_resolution():
	dim = 40
	res = 0.5
	gridsize=int(dim/res)
	representation="engA"
	projection="2D"

	# check projection is legal
	if projection not in ["2D", "3D"]:
		raise Exception("Projection is Illgal!")

	# check output
	myarray = image.cs_set_resolution(gridsize, representation, projection)
	if myarray.size == 0:
		raise Exception("Function is broken! No resolution to be set.")


def test_cs_coords_to_grid():
	mol = Chem.MolFromSmiles("CCN(CC)CCCl")
	mol, df_atom, df_bond, nancheckflag = image.cs_compute_features(mol)
	dim = 40
	res = 0.5

	# check output
	df_atom, atomcheckflag = image.cs_coords_to_grid(df_atom, dim, res)
	if df_atom.empty:
		raise Exception("Map coordinates to grid fails!")


def test_cs_check_grid_boundary():
	mol = Chem.MolFromSmiles("CCN(CC)CCCl")
	dim = 40
	res = 0.5
	gridsize=int(dim/res)
	mol, df_atom, df_bond, nancheckflag = image.cs_compute_features(mol)
	df_atom, atomcheckflag = image.cs_coords_to_grid(df_atom, dim, res)
	col_list = list(df_atom)
	# check if certain column name exist
	if 'x_scaled' not in col_list:
		raise Exception("x coordinates is not in the Atom dataframe!")
	if 'y_scaled' not in col_list:
		raise Exception("y coordinates is not in the Atom dataframe!")
	if 'z_scaled' not in col_list:
		raise Exception("z coordinates is not in the Atom dataframe!")
	
	# check output
	sizecheckflag = image.cs_check_grid_boundary(df_atom, gridsize)
	if sizecheckflag == None:
		raise Exception("None type sizecheckflag!")


def test_cs_channel_mapping():
	channel = image.cs_channel_mapping()
	if channel == {}:
		raise Exception("channel is empty! Initializing fails.")

	for i in np.arange(len(channel)):
		if channel[i][0] != i:
			raise Exception("channel set to wrong value!")


def test_cs_map_atom_to_grid():
	mol = Chem.MolFromSmiles("CCN(CC)CCCl")
	mol, df_atom, df_bond, nancheckflag = image.cs_compute_features(mol)
	dim = 40
	res = 0.5
	gridsize=int(dim/res)
	representation="engA"
	projection="2D"
	df_atom, atomcheckflag = image.cs_coords_to_grid(df_atom, dim, res)
	myarray = image.cs_set_resolution(gridsize, representation, projection)
	channel = image.cs_channel_mapping()

	if representation not in ['engA', 'engB', 'engC', 'engD', 'std']:
		raise Exception("No such representation setting available!")

	col_list = list(df_atom)
	check_list = ['idx', 'amu', 'pc', 'val_ex', 'val_im']
	for i in check_list:
		if i not in col_list:
			raise Exception("No column name ", i, " available!")

	# check output
	myarray = image.cs_map_atom_to_grid(myarray, channel, df_atom, projection, representation)
	if myarray.size == np.array([]).size:
		raise Exception("Function is broken! No attom mapping to grid!")


def test_unique_rows():
	a = np.arange(6).reshape(2,3)

	if type(a).__module__ != np.__name__:
		raise Exception("Wrong np.array type!")

	b = image.unique_rows(a)
	if not np.array_equal(a, b):
		raise Exception("Function alters the numpy array!")


def test_del_atom_on_line2D():
	dim = 40
	res = 0.5
	mol = Chem.MolFromSmiles("CCN(CC)CCCl")
	mol, df_atom, df_bond, nancheckflag = image.cs_compute_features(mol)
	df_atom, atomcheckflag = image.cs_coords_to_grid(df_atom, dim, res)
	leftatom = int(df_bond['atom1'][0])
	rightatom = int(df_bond['atom2'][0])
	lx_grid = int(df_atom['x_scaled'][leftatom])
	ly_grid = int(df_atom['y_scaled'][leftatom])
	rx_grid = int(df_atom['x_scaled'][rightatom])
	ry_grid = int(df_atom['y_scaled'][rightatom])

	num = 30
	bx = np.linspace(lx_grid, rx_grid, num)
	by = np.linspace(ly_grid, ry_grid, num)
	bx_trans = bx[np.newaxis].T
	by_trans = by[np.newaxis].T
	drawbond = np.concatenate((bx_trans,by_trans),axis=1)
	r_drawbond = np.around(drawbond)
	ur_drawbond = image.unique_rows(r_drawbond)

	final_drawbond = image.del_atom_on_line2D(ur_drawbond, lx_grid, ly_grid, rx_grid, ry_grid)
	if final_drawbond.size == np.array([]).size:
		raise Exception("All of data has been deleted!")


def test_cs_map_bond_to_grid():
	mol = Chem.MolFromSmiles("CCN(CC)CCCl")
	mol, df_atom, df_bond, nancheckflag = image.cs_compute_features(mol)
	dim = 40
	res = 0.5
	gridsize=int(dim/res)
	representation="engA"
	projection="2D"
	df_atom, atomcheckflag = image.cs_coords_to_grid(df_atom, dim, res)
	myarray = image.cs_set_resolution(gridsize, representation, projection)
	channel = image.cs_channel_mapping()
	myarray = image.cs_map_atom_to_grid(myarray, channel, df_atom, projection, representation)
	
	if representation not in ['engA', 'engB', 'engC', 'engD', 'std']:
		raise Exception("No such representation setting available!")

	col_list = list(df_atom)
	check_list = ['idx', 'amu', 'pc', 'val_ex', 'val_im']
	for i in check_list:
		if i not in col_list:
			raise Exception("No column name ", i, " available!")

	if 'btype' not in list(df_bond):
		raise Exception("No column name 'btype' in bond dataframe!")

	# check output
	myarray = image.cs_map_bond_to_grid(myarray, channel, df_atom, df_bond, projection, representation)
	if myarray.size == np.array([]).size:
		raise Exception("Function is broken! No bond mapping to grid!")









