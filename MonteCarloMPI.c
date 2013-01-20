#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <mpi.h>
#define N 67108864
#define PI 3.14159265358979323846 // przyblizenie liczby pi do 20 miejsc po przecinku

int main(int argc, char **argv)
{
	int moje_id, liczba_proc;
	long zsumowane, i, k = 0;
	clock_t start, stop;
	float x, y, pi_proc_rown;

	start = clock();

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &moje_id);
	MPI_Comm_size(MPI_COMM_WORLD, &liczba_proc);
	MPI_Status status;

//	srand(moje_id*5);
        srand(time(NULL));

        for(i = (moje_id * N / liczba_proc) + 1; i <= (moje_id + 1) * N / liczba_proc; i++)
        {
                x = rand() / (float) RAND_MAX;
                y = rand() / (float) RAND_MAX;
                if((x * x + y * y) <= 1.0f) k++;
        }

	MPI_Reduce(&k, &zsumowane, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        pi_proc_rown = 4 * (float)zsumowane / N;
	
	stop = clock();

	if(moje_id == 0) 
	{
		printf("\tCzas liczenia PI rownolegle na procesorze: %.6f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);
		printf("\tWartosc PI wg CPU (sekw.): %.10f (blad: %.10f).\n", pi_proc_rown, pi_proc_rown - PI);
	}

	MPI_Finalize();

	return 0;
}
