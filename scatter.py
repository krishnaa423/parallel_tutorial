
from mpi4py import MPI
import numpy as np 

comm = MPI.COMM_WORLD

mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

# Set to some array before scatter operation.
array_size = 8
if mpi_rank==0: 
    a = np.arange(array_size, dtype='f8')
else:
    a = np.zeros(array_size, dtype='f8')
b = np.zeros(int(array_size/mpi_size), dtype='f8')

# Print arrays before the operation.  
print(f'Before scatter. Rank: {mpi_rank}, a: {a}, b: {b}')

# Scatter. 
comm.Scatter(a, b, root=0)

comm.Barrier()
print('')

# Print arrays after the operation. 
print(f'After scatter. Rank: {mpi_rank}, a: {a}, b: {b}') 
