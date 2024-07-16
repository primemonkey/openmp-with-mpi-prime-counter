#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include "numgen.c"
 
#define DATA 0
#define RESULT 1
#define FINISH 2
 
int main(int argc, char **argv)
{
  Args ins__args;
  parseArgs(&ins__args, &argc, argv);
 
  // set number of threads
  omp_set_num_threads(ins__args.n_thr);
 
  // program input argument
  long inputArgument = ins__args.arg;
 
  struct timeval ins__tstart, ins__tstop;
 
  int threadsupport;
  int myrank, nproc;
  unsigned long int *numbers;
 
  int batchSize = 100; 
 
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threadsupport);
 
  if (threadsupport < MPI_THREAD_FUNNELED)
  {
    printf("\nThe implementation does not support MPI_THREAD_FUNNELED, it supports level %d\n", threadsupport);
    MPI_Finalize();
    return -1;
  }
 
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
 
  if (!myrank)
  {
    gettimeofday(&ins__tstart, NULL);
    numbers = (unsigned long int *)malloc(inputArgument * sizeof(unsigned long int));
    numgen(inputArgument, numbers);
  }
 
  MPI_Request request;
  if (myrank == 0)
  {
    for (unsigned long i = 0; i < inputArgument; i += batchSize)
    {
      int destination_rank = i / batchSize % (nproc - 1) + 1;
      MPI_Isend(&numbers[i], batchSize, MPI_UNSIGNED_LONG, destination_rank, DATA, MPI_COMM_WORLD, &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
 
    for (int i = 1; i < nproc; i++)
    {
      MPI_Isend(NULL, 0, MPI_UNSIGNED_LONG, i, FINISH, MPI_COMM_WORLD, &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
 
    int total_primes = 0;
    for (int i = 1; i < nproc; i++)
    {
      int primes_from_slave;
      MPI_Irecv(&primes_from_slave, 1, MPI_INT, i, RESULT, MPI_COMM_WORLD, &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);
      total_primes += primes_from_slave;
    }
 
    printf("Total number of prime numbers: %d\n", total_primes);
  }
  else
  {
    int prime_count = 0;
    MPI_Status status;
    unsigned long int received_numbers[batchSize]; 
    int flag;
    do
    {
      MPI_Iprobe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
      if (flag)
      {
        if (status.MPI_TAG == DATA)
        {
          MPI_Irecv(received_numbers, batchSize, MPI_UNSIGNED_LONG, 0, DATA, MPI_COMM_WORLD, &request); 
 
#pragma omp parallel for reduction(+ : prime_count)
          for (int i = 0; i < batchSize; ++i) 
          {
            int is_prime = 1;
            for (unsigned long j = 2; j * j <= received_numbers[i]; ++j)
            {
              if (received_numbers[i] % j == 0)
              {
                is_prime = 0;
                break;
              }
            }
            if (is_prime)
            {
              prime_count++;
            }
          }
        }
      }
    } while (!flag || (flag && status.MPI_TAG != FINISH));
    MPI_Isend(&prime_count, 1, MPI_INT, 0, RESULT, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }
 
  if (!myrank)
  {
    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
  }
 
  MPI_Finalize();
}
