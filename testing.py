
#region: Variables.
array_size = int(10e6)
task_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
#endregion

#region: Modules.
import numpy as np 
from mpi4py import MPI 
import time 
from math import ceil
#endregion

#region: Functions.
def add_array(comm: MPI.Comm, array_size):

    if comm==MPI.COMM_NULL: return 0, 0, 0

    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    a = np.arange(array_size, dtype='f8')
    b = np.zeros(ceil(array_size/mpi_size))
    comm.Scatterv(a, b, root=0)
    start_time = time.time()
    local_sum = np.sum(b)
    global_sum = np.zeros(1)
    comm.Reduce(local_sum, global_sum, MPI.SUM, root=0)
    stop_time = time.time()
    elapsed_time = stop_time - start_time

    return global_sum[0], elapsed_time, mpi_size

def get_comm_array(comm: MPI.Comm, task_sizes: list):
    
    comm_array = []
    
    for task_size in task_sizes:
        group = comm.Get_group()    
        group = group.Incl(range(task_size))
        comm_temp = comm.Create(group)
        comm_array.append(comm_temp)
        
    return comm_array

def main():
    global array_size
    global task_sizes

    comm = MPI.COMM_WORLD
    
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()
    
    
    
    comm_array = get_comm_array(comm, task_sizes)
    
    ntasks = []
    sums = []
    times = []
    
    for comm_entry in comm_array:
        sum, elapsed_time, comm_size = add_array(comm_entry, array_size)
        if mpi_rank==0: ntasks.append(comm_size)
        if mpi_rank==0: times.append(elapsed_time)
        if mpi_rank==0: sums.append(sum)
        
    # Plot the sums vs time. 
    if mpi_rank==0:
        print(f'{"ntask":^20}{"time":^20}{"sum":^20}')
        print(f'{"-":-<60}')
        for ntask, sum, time in zip(ntasks, sums, times):
            print(f'{ntask:^15}     {time:^15.10e}     {sum:^15.10e}')
#endregion

#region: Classes.
#endregion

#region: Main.
if __name__=='__main__':
    main()
#endregion