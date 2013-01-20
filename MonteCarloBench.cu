#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

#define IL_BLOKOW 256
#define IL_WATKOW 256
#define IL_WEWN_TESTOW 1024

#define PI 3.14159265358979323846 // przyblizenie liczby pi do 20 miejsc po przecinku

// Cudowna wersja metody Monte Carlo
__global__ void cuda_monte_carlo(float *wyniki, curandState *stany) 
{
	unsigned int moje_id = threadIdx.x + blockDim.x * blockIdx.x;
	int i;
	long k = 0;
	float x, y;

	curand_init(moje_id*moje_id, moje_id, 0, &stany[moje_id]); // Inicjalizacja CURAND

	for(i = 0; i < IL_WEWN_TESTOW; i++)
	{
		x = curand_uniform(&stany[moje_id]);
		y = curand_uniform(&stany[moje_id]);
		if((x * x + y * y) <= 1.0f) k++;
	}

	wyniki[moje_id] = (4 * (float)k / IL_WEWN_TESTOW);
}

// Sekwencyjna wersja na procesorze
float proc_sekw_monte_carlo(long ilosc_testow) 
{
	long k, i;
	float x, y;
	srand(time(NULL));

	for(i = 0; i < ilosc_testow; i++) 
	{
		x = rand() / (float) RAND_MAX;
		y = rand() / (float) RAND_MAX;
		if((x * x + y * y) <= 1.0f) k++;
	}

	return (4 * (float)k / ilosc_testow);
}

// Funkcja main
int main(int argc, char *argv[])
{
	int i;
	clock_t start, stop;
	float *gfx_w;
	curandState *gfx_stany;
	float proc[IL_BLOKOW * IL_WATKOW];
	
	printf("\t-> Ilosc blokow: %d.\n\t-> Ilosc watkow na blok: %d.\n\t-> Ilosc testow dla kazdego watku: %d.\n\n", IL_BLOKOW, IL_WATKOW, IL_WEWN_TESTOW);
	
	/*****************************************
        * Start wersji na CUDA
        *****************************************/
	
	start = clock();
	
	cudaMalloc((void **)&gfx_w, IL_WATKOW * IL_BLOKOW * sizeof(float));
	cudaMalloc((void **)&gfx_stany, IL_WATKOW * IL_BLOKOW * sizeof(curandState));
	
	cuda_monte_carlo<<<IL_BLOKOW, IL_WATKOW>>>(gfx_w, gfx_stany);
	cudaMemcpy(proc, gfx_w, IL_WATKOW * IL_BLOKOW * sizeof(float), cudaMemcpyDeviceToHost);
	
	float pi_cuda;
	
	for(i = 0; i < IL_WATKOW * IL_BLOKOW; i++)
	{
		pi_cuda += proc[i];
	}
	
	pi_cuda /= (IL_WATKOW * IL_BLOKOW);
	
	stop = clock();
	
	printf("\t-> Czas liczenia PI na CUDA: %.6f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);
	printf("\t-> Wartosc PI wg CUDA: %.10f (blad wzgledem rzeczywistej wartosci: %.10f).\n", pi_cuda, pi_cuda - PI);

	// ----------- Koniec wersji na CUDA
	
	/*****************************************
        * Start wersji sekwencyjnej na procesorze
        *****************************************/
        
	start = clock();
	
	float pi_proc_sekw = proc_sekw_monte_carlo(IL_WATKOW * IL_BLOKOW * IL_WEWN_TESTOW);
	
	stop = clock();
	
	printf("\t-> Czas liczenia PI sekwencyjnie na procesorze: %.6f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);
	printf("\t-> Wartosc PI wg CPU (sekw.): %.10f (blad wzgledem rzeczywistej wartosci: %.10f).\n", pi_proc_sekw, pi_proc_sekw - PI);

	// ----------- Koniec wersji sekwensyjnej na procesorze
	
	return 0;
}
