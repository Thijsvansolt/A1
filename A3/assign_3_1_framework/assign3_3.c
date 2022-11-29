/*Name Thijs van Solt, Fedja Matti
* Student ID: 13967681, 13953699
* This file contains a broadcast function for a ring topology
* A process can send a message using MPI to all other processes
* To run this file use make broadcast and make runbroadcast
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

// this function is the module oparation
int mod(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

// This function broadcasts a message to all other processes
// It stats with the root process and then sends the message to its left and right neighbors
// These processes send the messages to  there own left and right neighbors.
int MYMPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    MPI_Status status;
    int num_msg = 0;

    if (root == 0) {
        int left = mod(root - 1, size);
        int right = mod(root + 1, size);
        MPI_Send(buffer, count, datatype, left, 1 , comm);
        MPI_Send(buffer, count, datatype, right, 1, comm);
    }
    else{
        MPI_Recv(buffer, strlen(buffer) + 1, datatype, mod(root - 1, size), 1, comm, MPI_STATUS_IGNORE);
        MPI_Send(buffer, strlen(buffer) + 1, datatype, mod(root + 1, size) , 1, comm);
        MPI_Recv(buffer, strlen(buffer) + 1, datatype, mod(root + 1, size), 1, comm, MPI_STATUS_IGNORE);
        MPI_Send(buffer, strlen(buffer) + 1, datatype, mod(root - 1, size), 1, comm);
        printf("Broadcast msg is %s\n", (char *)buffer);
    }
}

// This function sets up the situation for the broadcast.
// It creates a communicator and then calls the broadcast function.
// It creates a message for the root process to broadcast.
// and a filler message for the other processes.
int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char msg[20];

    if (rank == 0) {
        strcpy(msg, "Hello world!");
        MYMPI_Bcast(msg, strlen(msg) + 1, MPI_CHAR, rank,  MPI_COMM_WORLD);
    } else {
        strcpy(msg, "filler message");
        MYMPI_Bcast(msg, strlen(msg) + 1, MPI_CHAR, rank,  MPI_COMM_WORLD);
    }

    MPI_Finalize();
}