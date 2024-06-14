
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
c = np.zeros(array_size, dtype='f8')
comm.Scatter(a, b, root=0)

# Print arrays before the operation.  
print(f'Before gather. Rank: {mpi_rank}, b: {b}, c: {c}')

# Gather. 
comm.Gather(b, c, root=0)

comm.Barrier()
print('')

# Print arrays after the operation. 
print(f'After gather. Rank: {mpi_rank}, b: {b}, c: {c}') 
