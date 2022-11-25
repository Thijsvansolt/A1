#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

int mod(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

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
        printf("final msg is %s\n", (char *)buffer);
    }
}

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char msg[4];

    if (rank == 0) {
        strcpy(msg, "Hoi");
        MYMPI_Bcast(msg, strlen(msg) + 1, MPI_CHAR, rank,  MPI_COMM_WORLD);
    } else {
        strcpy(msg, "Hai");
        MYMPI_Bcast(msg, strlen(msg) + 1, MPI_CHAR, rank,  MPI_COMM_WORLD);
    }


    MPI_Finalize();
}