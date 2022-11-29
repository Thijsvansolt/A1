/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "simulate.h"

#define TAG_COLLECT 3

int c = 0.15;

/* Add any global variables you may need. */

/* Add any functions you may need (like a worker) here. */

double *calculate(const int i_max, const int t_max, double *old_array,
                  double *current_array, double *next_array)
{
    for (int t = 0; t < t_max; t++)
    {
        for (int i = 0; i < i_max; i++)
        {
            next_array[i] = 2 * current_array[i] - old_array[i] + c * (current_array[i - 1] - (2 * current_array[i] - current_array[i + 1]));
        }
        *old_array = *current_array;
        *current_array = *next_array;
    }
    return current_array;
}

/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, double *old_array,
                 double *current_array, double *next_array)
{
    int rc, num_tasks, rank;
    rc = MPI_Init(NULL, NULL);

    if (rc != MPI_SUCCESS)
    {
        fprintf(stderr, "Unable to set up MPI ");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int workload = i_max / num_tasks;

    int left_neighbour = rank - 1;
    int right_neighbour = rank + 1;

    double prev[workload + 2];
    double cur[workload + 2];
    double next[workload + 2];


    // send to self
    for (int j = 0; j < workload; j++)
    {
        cur[j+1] = current_array[rank*workload+j];
        prev[j+1] = old_array[rank*workload+j];
    }

    // calculate values:
    for (int t = 0; t < t_max; t++)
    {
        if (rank == num_tasks - 1)
        {
            MPI_Send(&cur[1], 1, MPI_DOUBLE, left_neighbour, 1, MPI_COMM_WORLD);
            MPI_Recv(&cur[0], 1, MPI_DOUBLE, left_neighbour, 1, MPI_COMM_WORLD, &status);
        }
        else if (rank == 0)
        {
            MPI_Send(&cur[workload], 1, MPI_DOUBLE, right_neighbour, 1, MPI_COMM_WORLD);
            MPI_Recv(&cur[workload + 1], 1, MPI_DOUBLE, right_neighbour, 1, MPI_COMM_WORLD, &status);
        }
        else
        {
            MPI_Send(&cur[workload], 1, MPI_DOUBLE, right_neighbour, 1, MPI_COMM_WORLD);
            MPI_Send(&cur[1], 1, MPI_DOUBLE, left_neighbour, 1, MPI_COMM_WORLD);
            MPI_Recv(&cur[workload + 1], 1, MPI_DOUBLE, right_neighbour, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&cur[0], 1, MPI_DOUBLE, left_neighbour, 1, MPI_COMM_WORLD, &status);
        }

        for (int i = 1; i < workload - 1; i++)
        {
            next[i] = 2 * cur[i] - prev[i] + c * (cur[i - 1] - (2 * cur[i] - cur[i + 1]));
        }
        *prev = *cur;
        *cur = *next;
    }

    // for the workers:
    if (rank != 0)
    {
        MPI_Send(&cur[1], workload, MPI_DOUBLE, 0, TAG_COLLECT, MPI_COMM_WORLD);
        MPI_Finalize();
        exit(0);
    }

    // for the master:
    else
    {
        for (int i = 0; i < workload; i++)
        {
            current_array[i] = cur[i+1];
        }

        for (int t = 1; t < num_tasks; t++)
        {
            MPI_Recv(&current_array[t*workload], workload, MPI_DOUBLE, t, TAG_COLLECT, MPI_COMM_WORLD, &status);
        }
        MPI_Finalize();
        return current_array;
    }
    return current_array;
}
