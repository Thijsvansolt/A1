/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "simulate.h"

int c = 0.15;

/* Add any global variables you may need. */

/* Add any functions you may need (like a worker) here. */

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
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    for (int t = 0; t < t_max; t++)
    {
        for (int i = 0; i < i_max; i++)
        {
            next_array[i] = 2 * current_array[i] -old_array[i] + c * (current_array[i-1] - (2 * current_array[i] - current_array[i+1]));
        }
        *old_array = *current_array;
        *current_array = *next_array;
    }



    MPI_Finalize();

    return current_array;
}