
from mpi4py import MPI
import numpy as np 

comm = MPI.COMM_WORLD

mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

# Set array to zero initially. 
array_size = 4
a = np.zeros(shape=(array_size,))

# Set to some array before broadcast. 
if mpi_rank==0: 
    a = np.arange(array_size).astype(dtype='f8')
    

# Print arrays before the operation.  
print(f'Before broadcast. Rank: {mpi_rank}, a: {a}')

# Broadcast. 
comm.Bcast([a, MPI.DOUBLE], root=0)

comm.Barrier()
print('')

# Print arrays after the operation. 
print(f'After broadcast. Rank: {mpi_rank}, a: {a}') 
