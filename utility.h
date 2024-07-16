#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef struct InputArgs
{
  long arg;
  int n_thr;
  char marker[64];
} Args;

void parseArgs(Args* aptr, int* argc, char** argv)
{
  if (*argc < 4)
  {
    fprintf(stderr, "[Error] To few arguments!\nUsage:\n%s [a] [marker] [n_thr]\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  else
  {
    aptr->arg = atoll(argv[1]);
    memset((aptr->marker), 0, 64);
    strncpy((aptr->marker), argv[2], 63);
    aptr->n_thr = atoi(argv[3]);
  }
  *argc = 1;
  return;
}

#ifdef CUDA
__host__
#endif
void ins__printtime(struct timeval *start, struct timeval *stop, char *marker) {

  long time=1000000*(stop->tv_sec-start->tv_sec)+stop->tv_usec-start->tv_usec;
  printf("\n%s: Execution time = %ld microseconds\n", marker, time);

  return;
}

#endif