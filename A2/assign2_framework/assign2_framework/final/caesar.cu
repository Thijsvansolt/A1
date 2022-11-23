/*
 * Names: Thijs van Solt, Fedja Matti
 * Student IDS: 13967681, 13953699
 * BSc Computer Science UvA
 * Description: This file contains an GPU version for
 *              encryption and decryption using
                Caesar and Vigenere.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include<ctype.h>

#include "file.hh"
#include "timer.hh"

using namespace std;

__constant__ int file_size;
__constant__ int length_key;

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

// Encrypts a single character using Caesar
__global__ void encryptKernel(char* deviceDataIn, char* deviceDataOut, int* key) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < file_size) {
        deviceDataOut[i] = (deviceDataIn[i] + key[i % length_key]) % 256;
    }
}

// Decrypts a single character using Caesar
__global__ void decryptKernel(char* deviceDataIn, char* deviceDataOut, int* key) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < file_size) {
        deviceDataOut[i] = (deviceDataIn[i] - key[i % length_key]) % 256;
    }
}

// Encrypts a file using Caesar or Vigenere
int EncryptSeq (int n, char* data_in, char* data_out, int key_length, int *key) {
    timer sequentialTime = timer("Sequential encryption");

    sequentialTime.start();
    for (int i = 0; i < n; i++) {
        data_out[i] = (data_in[i] + key[i % key_length]) % 256;
    }
    sequentialTime.stop();

    cout << fixed << setprecision(6);
    cout << "Encryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

    return 0;
}

// Decrypts a file using Caesar or Vigenere
int DecryptSeq (int n, char* data_in, char* data_out, int key_length, int *key)
{
    timer sequentialTime = timer("Sequential decryption");

    sequentialTime.start();
    for (int i = 0; i < n; i++) {
        data_out[i] = (data_in[i] - key[i % key_length]) % 256;
    }
    sequentialTime.stop();

    cout << fixed << setprecision(6);
    cout << "Decryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

    return 0;
}

/* Wrapper for your encrypt kernel, i.e., does the necessary preparations and
 * calls your kernel. */
int EncryptCuda (int n, char* data_in, char* data_out, int key_length, int *key) {
    int threadBlockSize = 512;
    checkCudaCall(cudaMemcpyToSymbol(length_key, &key_length, sizeof(int)));
    checkCudaCall(cudaMemcpyToSymbol(file_size, &n, sizeof(int)));

    int* deviceKey = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceKey, key_length * sizeof(int)));
    if (deviceKey == NULL) {
        cerr << "Error allocating device memory for key" << endl;
        exit(EXIT_FAILURE);
    }
    // allocate the vectors on the GPU
    char* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        checkCudaCall(cudaFree(deviceKey));
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char* deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceKey, key, key_length*sizeof(int), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    if (n % threadBlockSize == 0) {
        encryptKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceDataIn, deviceDataOut, deviceKey);
    } else {
        encryptKernel<<<(n/threadBlockSize) + 1, threadBlockSize>>>(deviceDataIn, deviceDataOut, deviceKey);
    }
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));
    checkCudaCall(cudaFree(deviceKey));

    cout << fixed << setprecision(6);
    cout << "Encrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Encrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

   return 0;
}

/* Wrapper for your decrypt kernel, i.e., does the necessary preparations and
 * calls your kernel. */
int DecryptCuda (int n, char* data_in, char* data_out, int key_length, int *key) {
    int threadBlockSize = 512;
    checkCudaCall(cudaMemcpyToSymbol(length_key, &key_length, sizeof(int)));
    checkCudaCall(cudaMemcpyToSymbol(file_size, &n, sizeof(int)));

    int* deviceKey = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceKey, key_length * sizeof(int)));
    if (deviceKey == NULL) {
        cerr << "Error allocating device memory for key" << endl;
        exit(EXIT_FAILURE);
    }

    // allocate the vectors on the GPU
    char* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        checkCudaCall(cudaFree(deviceKey));
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char* deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceKey, key, key_length*sizeof(int), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    if (n % threadBlockSize == 0) {
        decryptKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceDataIn, deviceDataOut, deviceKey);
    } else {
        decryptKernel<<<(n/threadBlockSize) + 1, threadBlockSize>>>(deviceDataIn, deviceDataOut, deviceKey);
    }
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));
    checkCudaCall(cudaFree(deviceKey));

    cout << fixed << setprecision(6);
    cout << "Decrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Decrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

   return 0;
}

/* Entry point to the function! */
int main(int argc, char* argv[]) {
    // Check if there are enough arguments
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " key..." << endl;
        cout << " - key: one or more values for the encryption key, separated "
                "by spaces" << endl;
        return EXIT_FAILURE;
    }

    // Parse the keys from the command line arguments
    int key_length = argc - 1;
    int *enc_key = new int[key_length];
    for (int i = 0; i < key_length; i++) {
        enc_key[i] = atoi(argv[i + 1]);
    }

    // Check if the original.data file exists and what it's size is
    int n;
    n = fileSize("10mb.txt");
    if (n == -1) {
        cout << "File not found! Exiting ... " << endl;
        exit(0);
    }


    // Read the file in memory from the disk
    char* data_in = new char[n];
    char* data_out = new char[n];
    readData("10mb.txt", data_in);

    cout << "Encrypting a file of " << n << " characters." << endl;

    EncryptSeq(n, data_in, data_out, key_length, enc_key);
    writeData(n, "sequential.data", data_out);
    EncryptCuda(n, data_in, data_out, key_length, enc_key);
    writeData(n, "cuda.data", data_out);

    readData("cuda.data", data_in);

    cout << "Decrypting a file of " << n << "characters" << endl;
    DecryptSeq(n, data_in, data_out, key_length, enc_key);
    writeData(n, "sequential_recovered.data", data_out);
    DecryptCuda(n, data_in, data_out, key_length, enc_key);
    writeData(n, "recovered.data", data_out);

    delete[] data_in;
    delete[] data_out;

    return 0;
}
