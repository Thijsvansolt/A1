#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

size_t size = 10;

typedef struct queue
{
    int *data;
    int front;
    int rear;
    size_t size;
    size_t max_size;
} queue;

typedef struct args
{
    queue *input;
    queue *output;
    int filter_value;
} args;

queue *init(size_t size)
{
    queue *q = malloc(sizeof(queue));
    if (q == NULL)
        return NULL;
    q->data = malloc(size * sizeof(int));
    if (q->data == NULL)
    {
        free(q);
        return NULL;
    }
    q->front = 0;
    q->rear = 0;
    q->size = 0;
    q->max_size = size;

    return q;
}

int push(queue *q, int value)
{
    if (q->size < q->max_size)
    {
        q->data[q->rear % q->max_size] = value;
        q->rear++;
        q->size++;
        return 0;
    }
    else
        return -1;
}

int pop(queue *q)
{
    if (q->size > 0)
    {
        int result = q->data[q->front];
        q->front++;
        q->size--;
        return result;
    }
    else
        return -1;
}

void *filter(void *a)
{
    args *ar = (args *)a;

    queue *input = ar->input;

    queue *output = ar->output;

    queue *new_input = NULL;

    for (int i = 1; i < 11; i++)
    {
        if (input->data[i] % ar->filter_value != 0)
        {
            if (!new_input)
            {
                new_input = init(size);
                push(new_input, input->data[i]);
                args *new_a = malloc(sizeof(args));
                new_a->input = new_input;
                new_a->output = init(size);
                new_a->filter_value = input->data[i];
                push(output, input->data[i]);
                pthread_t new_filter = (pthread_t*) malloc(sizeof(pthread_t)*1);
                pthread_create(&new_filter, NULL, filter, (void *)new_a);
                pthread_join(new_filter, NULL);
            }
            else{
                push(new_input, input->data[i]);
            }
        }
    }
}

int main(void)
{
    queue *input = init(10);
    queue *output = init(100);

    for (int i = 0; i < 10; i++)
    {
        input->data[i] = i;
    }

    args *a = malloc(sizeof(args));
    a->input = input;
    a->output = output;
    a->filter_value = 2;

    pthread_t test;
    pthread_create(&test, NULL, filter, (void *)a);
    pthread_join(test, NULL);

    for (int i = 0; i <= 10; i++)
    {
        int value = pop(output);
        printf("%d\n", value);
    }

    return 0;
}