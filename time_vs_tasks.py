
# Some variables. 
array_size = 100
task_sizes = [1, 2, 3, 4, 5, 6, 7, 8]


from mpi4py import MPI 
import numpy as np 
import time 

def add_array(array_size, comm: MPI.Comm):
    
    if comm==MPI.COMM_NULL: return 0, 0
    
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()
    
    if mpi_rank==0: 
        a = np.arange(array_size, dtype='f8')
    else:
        a = np.zeros(array_size, dtype='f8')
    b = np.zeros(array_size, dtype='f8')
    c = np.zeros(array_size, dtype='f8')
    
    start_time = time.time()
    comm.Scatterv(a, b, root=0)
    comm.Allreduce(b, c, op=MPI.SUM)
    
    sum = np.sum(c)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
        
    return sum, elapsed_time

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
        sum, elapsed_time = add_array(array_size, comm_entry)
        if mpi_rank==0: ntasks.append(comm_entry.Get_size())
        if mpi_rank==0: times.append(elapsed_time)
        if mpi_rank==0: sums.append(sum)
        
    # Plot the sums vs time. 
    if mpi_rank==0:
        for ntask, sum, time in zip(ntasks, sums, times):
            print(f'ntask: {ntask}, time: {time}, sum: {sum}')
    
    
if __name__=='__main__':
    main()
