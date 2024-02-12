# Read coefficients from txt files

import os.path
import sys
import numpy as np

# Import numerical solver
from F2DHIT_DSMAG import *

SMAG_solver(0.12, 'SMAG')


'''
# Read iteration number
iteration = sys.argv[1]
print(iteration)



# print(os.path.dirname(__file__))
contents = []
with open(os.path.dirname(__file__) + '/../versions_' + iteration + '.txt') as f:
    contents = f.read().splitlines()
  

N_ens = len(contents) 
CS = np.zeros([N_ens])
versions = np.zeros([N_ens])

for n_ens in range(N_ens):
    
    # Obtain SMAG coefficients
    with open(os.path.dirname(__file__) + '/../param_defs/' + contents[n_ens]) as f:
        coeffs = f.read().splitlines()
        coeffs = coeffs[0].split()
        CS[n_ens] = float(coeffs[1])

    # Obtain version names
    version_ = contents[n_ens].replace("_"," ")
    version_ = version_.replace("."," ")
    version_ = [int(s) for s in version_.split() if s.isdigit()]
    versions[n_ens] = version_[0]

    # Run SMAG numerical solver
    SMAG_solver(float(coeffs[1]), version_[0])

print(versions)
print(CS)

with open(os.path.dirname(__file__) + '/version_' + iteration + '.txt', 'w') as f:
    for line in range(len(CS)):
	    f.write(str(CS[line]))
	    f.write('\n')
'''