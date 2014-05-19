// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "driver_types.h"

#include <stdio.h>


__global__ void kernel(long long * numElements)
{
	long long i = 0;
	while(i < 1000000000)
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

	err = cudaMemcpy(d_numElements, h_numElements, sizeof(long long), cudaMemcpyHostToDevice);

	kernel<<<1, 16>>>(d_numElements);

	//Хендл event'а
	cudaEvent_t syncEvent;

	cudaEventCreate(& syncEvent);		//Создаем event
	cudaEventRecord(syncEvent, 0);		//Записываем event
	cudaEventSynchronize(syncEvent);	//Синхронизируем event

	err = cudaMemcpy(h_numElements, d_numElements, sizeof(long long), cudaMemcpyDeviceToHost);

	printf("[%d elements]\n", *h_numElements);

	system("pause");

	return 0;

}
