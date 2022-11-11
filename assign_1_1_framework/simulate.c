/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "simulate.h"


/* Add any global variables you may need. */

struct all_threads {
    double *old;
    double *current;
    double *new;
    double c;
    int i_max;
};

/* Add any functions you may need (like a worker) here. */

// 1-dimensional wave equation function
void *wave_eq(void *all_info){
    struct all_threads *all = (struct all_threads *) all_info;
    for(int i = 0; i < all->i_max - 1; i++){
        all->new[i] = 2 * (all->current[i] - all->old[i]) + all->c*(all->old[i - 1] - (2 * (all->current[i] - all->current[i + 1])));
    }
//     return all->new;
    // printf("%f\n", all->c);
    // for (int i = 0; i < all->i_max; i++){
    //     printf("%f\n", all->new[i]);
    // }
    return 0;
}

/*
 * Executes the entire simulation.
 *
 * Implement your code here.
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
    struct all_threads *arrays = (struct all_threads *)malloc(sizeof(struct all_threads));
    arrays->old = old_array;
    arrays->current = current_array;
    arrays->new = next_array;
    arrays->c = 0.15;
    arrays->i_max = i_max;



    /*
     * After each timestep, you should swap the buffers around. Watch out none
     * of the threads actually use the buffers at that time.
     */
    int number = num_threads;
    pthread_t *thd;
    thd = (pthread_t*)malloc(sizeof(pthread_t)*number);

    for (int i = 0; i < number; i++){
        pthread_create(&thd[i], NULL, wave_eq, (void *)arrays);
    }
    for (int i = 0; i < number; i++){
        pthread_join(thd[i], NULL);
    }
    // exit(0);


    /* You should return a pointer to the array with the final results. */
    return current_array;
    
}



