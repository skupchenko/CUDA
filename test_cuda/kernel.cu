// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "driver_types.h"

#include <stdio.h>


__global__ void kernel(long long * numElements)
{
	long long i = 0;
	while(i < 48000000)
	{
		++i;
	}
	*numElements = i;
}

int main(void)
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Allocate the host input numElements
	long long * h_numElements = (long long *)malloc(sizeof(long long));

	* h_numElements = 0;

	// Allocate the device input numElements
	long long * d_numElements = NULL;
	err = cudaMalloc((void **) & d_numElements, sizeof(long long));
	
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device numElements (error code %s)!\n", cudaGetErrorString(err));
		system("pause");
		exit(EXIT_FAILURE);
	}

	// Copy the host h_numElements in host memory to the device input d_numElements in
	// device memory
	err = cudaMemcpy(d_numElements, h_numElements, sizeof(long long), cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy h_numElements from host to device (error code %s)!\n", cudaGetErrorString(err));
		system("pause");
		exit(EXIT_FAILURE);
	}

	kernel<<<1, 16>>>(d_numElements);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		system("pause");
		exit(EXIT_FAILURE);
	}

	//Handle event'à
	cudaEvent_t syncEvent;

	cudaEventCreate(& syncEvent);		//Create event
	cudaEventRecord(syncEvent, 0);		//Record event
	cudaEventSynchronize(syncEvent);	//Synchronize event

	// Copy the device result in device memory to the host result
	// in host memory.
	err = cudaMemcpy(h_numElements, d_numElements, sizeof(long long), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy numElements from device to host (error code %s)!\n", cudaGetErrorString(err));
		system("pause");
		exit(EXIT_FAILURE);
	}

	printf("[%d elements]\n", *h_numElements);

	// Free device global memory
	err = cudaFree(d_numElements);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device d_numElements (error code %s)!\n", cudaGetErrorString(err));
		system("pause");
		exit(EXIT_FAILURE);
	}

	// Free host memory
	free(h_numElements);

	// Reset the device and exit
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	err = cudaDeviceReset();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
		system("pause");
		exit(EXIT_FAILURE);
	}

	system("pause");

	return 0;
}
