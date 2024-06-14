
from mpi4py import MPI

comm = MPI.COMM_WORLD

mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

print(f'Rank: {mpi_rank}, size: {mpi_size}')
