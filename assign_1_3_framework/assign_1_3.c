#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <math.h>



void *SieveOfEratosthenes(void *n) {
    int num = *(int *)n;
    bool prime[num + 1];
    memset(prime, true, sizeof(prime));

    for (int p = 2; p * p <= num; p++) {
        if (prime[p] == true) {
            for (int i = p * p; i <= num; i += p) {
                prime[i] = false;
            }
        }
    }

    // Print all prime numbers
    for (int p = 2; p <= num; p++) {
        if (prime[p]) {
            printf("%d\n",p);
        }
    }
    return NULL;
}

int *make_ranges(int num_threads, int i_max){
    int *ranges = malloc(sizeof(int) * (2 * num_threads));
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

int main(int argc, char *argv[]) {
    int num_threads = atoi(argv[1]);
    int n = atoi(argv[2]);

    pthread_t *thd;
    thd = (pthread_t*)malloc(sizeof(pthread_t)*num_threads);
    for (int i = 0; i < num_threads; i++){
        pthread_create(&thd[i], NULL, SieveOfEratosthenes, &n);
    }
    for (int i = 0; i < num_threads; i++){
        pthread_join(thd[i], NULL);
    }
    // SieveOfEratosthenes(n);
    return 0;
}