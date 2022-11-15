/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#include "simulate.h"

float c = 0.15;

/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old_array, double *current_array, double *next_array)
{
    int workload = t_max / num_threads;

    #pragma omp parallel num_threads(num_threads) reduction (+:workload)
    {
        for (int t = 0; t < t_max / num_threads; t++)
        {
            for (int i = 0; i < i_max; i++)
            {
                next_array[i] = 2 * current_array[i] -old_array[i] + c * (current_array[i-1] - (2 * current_array[i] - current_array[i+1]));
            }

        }
    }

    return current_array;
}
