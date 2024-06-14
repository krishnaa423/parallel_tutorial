
from mpi4py import MPI
import numpy as np 

comm = MPI.COMM_WORLD

mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

# Set array to zero initially. 
array_size = 4
a = np.zeros(shape=(array_size,))

# Set to some array before sending and receiving. 
if mpi_rank==0: 
    a = np.arange(array_size).astype(dtype='f8')
    

# Print arrays before the operation.  
print(f'Before send and receive. Rank: {mpi_rank}, a: {a}')

# Send and Receive. 
if mpi_rank==0:
    comm.Isend([a, MPI.DOUBLE], dest=1, tag=10)
elif mpi_rank==1:
    req = comm.Irecv([a, MPI.DOUBLE], source=0, tag=10)


# Print arrays after the operation. 
print(f'After send and receive. Rank: {mpi_rank}, a: {a}') 

# Print arrays after waiting. 
if mpi_rank==1: req.Wait()

comm.Barrier()
print('')
print(f'After send and receive and waiting. Rank: {mpi_rank}, a: {a}') 
