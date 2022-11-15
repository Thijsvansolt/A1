/*
 * Names: Thijs van Solt, Fedja Matti
 * Student IDS: 13967681, 13953699
 * BSc Computer Science UvA
 * Description: This file contains an multi threaded wave-equation function.
 *             It uses the pthread library to create threads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

#include "simulate.h"


/* Add any global variables you may need. */
pthread_barrier_t barrier;

// This is the struct that contains the data for each thread and wave-equation.
struct all_threads {
    double *old;
    double *current;
    double *new;
    double c;
    int start;
    int end;
    int t_max;
};

/* Add any functions you may need (like a worker) here. */

// This function is the worker function for each thread.
// It calculates the wave-equation for each thread.
// It uses the struct all_threads to get the data.
// It uses the pthread_barrier_t to synchronize the threads.
void *wave_eq(void *all_info){
    struct all_threads *all = (struct all_threads *) all_info;
    for(int t = 1; t <= all->t_max - 1; t++){
        for(int i = all->start; i <= all->end; i++) {
            all->new[i] = 2 * all->current[i] - all->old[i] + all->c *
            (all->current[i - 1] - (2 * all->current[i] - all->current[i + 1]));
        }
        pthread_barrier_wait(&barrier);
        double *temp = all->old;
        all->old = all->current;
        all->current = all->new;
        all->new = temp;
    }
    free(all_info);
    return NULL;
}

// This function determines the workload for each thread.
// It returns an array with the workload for each thread.
int *make_ranges(int num_threads, int i_max){
    int *ranges = malloc(sizeof(int) *(2 * num_threads));
    int range = i_max / num_threads;

    for (int i = 0; i < num_threads; i++) {
        if (i == 0){
            int start = 0;
            int end = (i + 1) * range;
            ranges[2 * i] = start;
            ranges[2 * i + 1] = end;
        } else {
            int start = i * range + 1;
            int end = (i + 1) * range;
            ranges[2 * i] = start;
            ranges[2 * i + 1] = end;
        }
    }
    return ranges;
}

/*
 * Executes the entire simulation.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use (excluding the main threads)
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old_array, double *current_array, double *next_array)
{
    pthread_barrier_init(&barrier, NULL, num_threads);
    int *ranges = make_ranges(num_threads, i_max);
    int number = num_threads;
    pthread_t *thd;
    thd = (pthread_t*)malloc(sizeof(pthread_t)*number);

    for (int i = 0; i < number; i++){
        struct all_threads *all_info = (struct all_threads *)malloc(sizeof(struct all_threads));
        all_info->old = old_array;
        all_info->current = current_array;
        all_info->new = next_array;
        all_info->c = 0.15;
        all_info->start = ranges[2 * i];
        all_info->end = ranges[2 * i + 1];
        all_info->t_max = t_max;
        pthread_create(&thd[i], NULL, wave_eq, all_info);
    }

    for (int i = 0; i < number; i++){
        pthread_join(thd[i], NULL);
    }

    free(ranges);
    free(thd);
    return current_array;
}



