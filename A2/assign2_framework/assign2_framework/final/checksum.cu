/*
 * Names: Thijs van Solt, Fedja Matti
 * Student IDS: 13967681, 13953699
 * BSc Computer Science UvA
 * Description: This file contains an GPU version for
 *              calculation the checksum.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

#include "timer.hh"
#include "file.hh"

using namespace std;
__constant__ int size_of_file;


/* Utility function, use to do error checking for CUDA calls
 *
 * Use this function like this:
 *     checkCudaCall(<cuda_call>);
 *
 * For example:
 *     checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
 *
 * Special case to check the result of the last kernel invocation:
 *     kernel<<<...>>>(...);
 *     checkCudaCall(cudaGetLastError());
**/
static void checkCudaCall(cudaError_t result) {
  if (result != cudaSuccess) {
      cerr << "cuda error: " << cudaGetErrorString(result) << endl;
      exit(EXIT_FAILURE);
  }
}

// Kernel function to calculate the checksum
// It divides the block in half and sums the second half with the first half
// It does this until the final sum is calculated.
 __global__ void checksumKernel(unsigned int* result, unsigned int *deviceDataIn){
    __shared__ unsigned int sdata[512];
    ;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i > size_of_file){
        sdata[threadIdx.x] = 0;
    }
    else{
        sdata[threadIdx.x] = deviceDataIn[i];
    }
    __syncthreads();
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
        }

    if (threadIdx.x == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// Function to calculate the checksum
unsigned int checksumSeq (int n, unsigned int* data_in) {
    int i;
    timer sequentialTime = timer("Sequential checksum");

    sequentialTime.start();
    unsigned int result = 0;
    for (i=0; i<n; i++) {
        result += data_in[i];
    }
    sequentialTime.stop();

    cout << fixed << setprecision(6);
    cout << "Checksum (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

    return result;
}

/**
 * The checksumCuda handler that initialises the arrays to be used and calls
 * the checksum kernel. It also computes the missing values not calculated
 * on the GPU. It then adds all values together and prints the checksum
 */
 unsigned int checksumCuda (int n, unsigned int* data_in) {
    int threadBlockSize = 512;
    checkCudaCall(cudaMemcpyToSymbol(size_of_file, &n, sizeof(int)));
    // Allocate the vectors & the result int on the GPU
    unsigned int* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(unsigned int)));
    if (deviceDataIn == NULL) {
        cout << "Could not allocate input data on GPU." << endl;
        exit(1);
    }
    unsigned int* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, sizeof(unsigned int)));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "Could not allocate result integer on GPU." << endl;
        exit(1);
    }

    timer kernelTime  = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // Copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(unsigned int), cudaMemcpyHostToDevice));
    memoryTime.stop();

    kernelTime.start();
    if (n % threadBlockSize == 0) {
        checksumKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceResult, deviceDataIn);
    } else {
        checksumKernel<<<(n/threadBlockSize) + 1, threadBlockSize>>>(deviceResult, deviceDataIn);
    }

    cudaDeviceSynchronize();
    kernelTime.stop();

    // Check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // Copies back the correct data
    unsigned int result;
    checkCudaCall(cudaMemcpy(&result, deviceResult, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Releases the GPU data
    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceResult));

    // The times are printed
    cout << fixed << setprecision(6);
    cout << "Kernel: \t\t" << kernelTime.getElapsed() << " seconds." << endl;
    cout << "Memory: \t\t" << memoryTime.getElapsed() << " seconds." << endl;

    return result;
}

/* Entry point to the program. */
int main(int argc, char* argv[]) {
    int n;
    char* mode;
    char* fileName;

    // Arg parse
    if (argc == 3) {
        fileName = argv[1];
        mode = argv[2];

        cout << "Running in '" << mode << "' mode" << endl;
        cout << "Opening file " << fileName << endl;
    } else {
        cout << "Usage: " << argv[0] << " filename mode" << endl;
        cout << " - filename: name of the file for which the checksum will be "
                "computed." << endl;
        cout << " - mode: one of the three modes for which the program can "
                "run." << endl;
        cout << "   Available options are:" << endl;
        cout << "    * seq: only runs the sequential implementation" << endl;
        cout << "    * cuda: only runs the parallelized implementation" << endl;
        cout << "    * both: runs both the sequential and the parallelized "
                "implementation" << endl;

        return EXIT_FAILURE;
    }
    n = fileSize(fileName);
    if (n == -1) {
        cerr << "File '" << fileName << "' not found" << endl;
        exit(EXIT_FAILURE);
    }

    char* data_in = new char[n];
    readData(fileName, data_in);
    unsigned int *data_in_raw = new unsigned int[n];
    for (int i = 0; i < n; i++){
        data_in_raw[i] = data_in[i];
    }

    /* Check the option to determine the functions to be called */
    if (strcmp(mode, "seq") == 0){
        // Only sequential checkusm is ran
        unsigned int checksum = checksumSeq(n, data_in_raw);
        cout << "Sequential checksum: " << checksum << endl;
    } else if (strcmp(mode, "cuda") == 0) {
        // Only cuda checksum is ran
        unsigned int checksum = checksumCuda(n, data_in_raw);
        cout << "CUDA checksum: " << checksum << endl;
    } else if (strcmp(mode, "both") == 0){
        // Both the sequential and the cuda checksum are run
        unsigned int checksum = checksumCuda(n, data_in_raw);
        cout << "CUDA checksum: " << checksum << endl;
        checksum = checksumSeq(n, data_in_raw);
        cout << "Sequential checksum: " << checksum << endl;
    } else {
        cerr << "Unknown mode '" << mode << "'; only accepts 'seq', 'cuda' or "
                "'both'" << endl;
        delete[] data_in;
        delete[] data_in_raw;
        exit(EXIT_FAILURE);
    }

    delete[] data_in;
    delete[] data_in_raw;
    return EXIT_SUCCESS;
}
