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

	//Хендл event'а
	cudaEvent_t syncEvent;

	cudaEventCreate(& syncEvent);		//Создаем event
	cudaEventRecord(syncEvent, 0);		//Записываем event
	cudaEventSynchronize(syncEvent);	//Синхронизируем event

	return 0;

}
