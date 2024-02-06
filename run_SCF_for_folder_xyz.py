## -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
from pyscf import gto, dft, lib, scf
import torch 
from torch.autograd import Variable 
import torch.nn as nn 
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
import ase
from ase.io import read
from ase import Atoms
from ase.collections import g2
import xc_pcNN as xc
import os
from jax import numpy as jnp
import sys
import glob
lib.num_threads(16)

def unrolling(w1, w2, w3, w4, w1c, w2c, w3c, w4c, b1, b2, b3, b4, b1c, b2c, b3c, b4c):
	# routine to flatten parameter matrices and put then in a single vector for saving and parsing
	
	w1=w1.flatten()
	w2=w2.flatten()
	w3=w3.flatten()
	w4=w4.flatten()
	w1c=w1c.flatten()
	w2c=w2c.flatten()
	w3c=w3c.flatten()
	w4c=w4c.flatten()
	b1=b1.flatten()
	b2=b2.flatten()
	b3=b3.flatten()
	b4=b4.flatten()
	b1c=b1c.flatten()
	b2c=b2c.flatten()
	b3c=b3c.flatten()
	b4c=b4c.flatten()
	return np.concatenate((w1, w2, w3, w4, w1c, w2c, w3c, w4c, b1, b2, b3, b4, b1c, b2c, b3c, b4c))
	
def rolling(hyperparameters):
	#Routine to split and reshape the hyperparameters vector in the matrices for use in the functionals 
	
	###  w1
	#print(hyperparameters.shape)
	start=end=0
	end=start+16*2
	w1=hyperparameters[start:end].reshape((16,2))
	###  w2
	start=end
	end=start+16*16
	w2=hyperparameters[start:end].reshape((16,16))
	###  w3
	start=end
	end=start+16*16
	w3=hyperparameters[start:end].reshape((16,16))
	###  w4
	start=end
	end=start+1*16
	w4=hyperparameters[start:end].reshape((1,16))
	###  w1c
	start=end
	end=start+16*4
	w1c=hyperparameters[start:end].reshape((16,4))
	###  w2c
	start=end
	end=start+16*16
	w2c=hyperparameters[start:end].reshape((16,16))
	###  w3c
	start=end
	end=start+16*32
	w3c=hyperparameters[start:end].reshape((16,32))
	###  w4c
	start=end
	end=start+1*16
	w4c=hyperparameters[start:end].reshape((1,16))
	
	###  b1
	start=end
	end=start+16
	b1=hyperparameters[start:end].reshape((16,))
	###  b2
	start=end
	end=start+16
	b2=hyperparameters[start:end].reshape((16,))
	###  b3
	start=end
	end=start+16
	b3=hyperparameters[start:end].reshape((16,))
	###  b4
	start=end
	end=start+1
	b4=hyperparameters[start:end].reshape((1,))
	###  b1c
	start=end
	end=start+16
	b1c=hyperparameters[start:end].reshape((16,))
	###  b2c
	start=end
	end=start+16
	b2c=hyperparameters[start:end].reshape((16,))
	###  b3c
	start=end
	end=start+16
	b3c=hyperparameters[start:end].reshape((16,))
	###  b4c
	start=end
	end=start+1
	b4c=hyperparameters[start:end].reshape((1,))
	
	if end != len(hyperparameters):
		print("Something is out of place... ")
		print("end=",end," but we expected it to be ", len(hyperparameters))
		exit()
	
	return w1, w2, w3, w4, w1c, w2c, w3c, w4c, b1, b2, b3, b4, b1c, b2c, b3c, b4c




if len(sys.argv)>2 :
	print("ERROR --->  incorrect number of arguments")
	print("")
	print("Exemple: python run_g2_dft.py  <parameterfile>.npy")
	exit()
hidden=16
model = xc.Net()
model.hidden = hidden
model.mkmat(seed=172)


file_par=str(sys.argv[1])
print("Reading parameters for the ANN from file: ", file_par)
hyperparam = np.load(file_par)
w1, w2, w3, w4, w1c, w2c, w3c, w4c, b1, b2, b3, b4, b1c, b2c, b3c, b4c = rolling(hyperparam)

model.w1 = jnp.array(w1)
model.w2 = jnp.array(w2)
model.w3 = jnp.array(w3)
model.w4 = jnp.array(w4)
model.w1c = jnp.array(w1c)
model.w2c = jnp.array(w2c)
model.w3c = jnp.array(w3c)
model.w4c = jnp.array(w4c)
model.b1 = jnp.array(b1)
model.b2 = jnp.array(b2)
model.b3 = jnp.array(b3)
model.b4 = jnp.array(b4)
model.b1c = jnp.array(b1c)
model.b2c = jnp.array(b2c)
model.b3c = jnp.array(b3c)
model.b4c = jnp.array(b4c)

for file in glob.glob("xyz/*.xyz"):
	print("Reading geometry from file: ", file)
	name=file.split("/")[1]
	name=name.split(".")[0]

	mol = gto.Mole()
	mol.verbose = 5
	mol.atom    = file
	with open(file) as f:
		lines = f.readlines()
	charge=int(lines[1].split()[1])
	spin=int(lines[1].split()[2])-1
	mol.charge=charge
	mol.spin=spin
	mol.basis   = "ccpcvtz"
	mol.output = 'output'+name+'.log'
	mol.build()


	# HF calculation #
	if mol.spin==0:
		mf = scf.RHF(mol)
	else:
		mf = scf.UHF(mol)
	#mf = scf.UHF(mol)
	mf.kernel()
	dm1 = mf.make_rdm1()
	
	
	# DFT calculation #
	if mol.spin==0:
		mfl = dft.RKS(mol)
	else:
		mfl = dft.UKS(mol)
	#mfl = dft.UKS(mol)
	mfl = mfl.define_xc_(model.eval_xc, 'MGGA')
	mfl.damp = 0.5
	mfl.diis_start_cycle = 50
	mfl.conv_tol=1e-8
	mfl.max_cycle=500
	#mfl.init_guess = 'minao'
	mfl.kernel(dm0=dm1)
	print(name+' DFT-xcnn total energy = ', mfl.e_tot)
	print('-------------------------------------------------------------------------------------------')
	
	mfl.analyze()
	
	


