__global__ void kernel()
{
	long long i = 0;
	while(i < 10000000000)
	{
		++i;
	}
}

int main(void)
{

	kernel<<<1, 16>>>();

	//����� event'�
	cudaEvent_t syncEvent;

	cudaEventCreate(& syncEvent);		//������� event
	cudaEventRecord(syncEvent, 0);		//���������� event
	cudaEventSynchronize(syncEvent);	//�������������� event

	return 0;

}
