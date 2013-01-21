all:
	mpicc MonteCarloMPI.c -lrt -o MonteCarloMPI
	nvcc MonteCarloBench.cu -o MonteCarloBench
clean:
	rm -f MonteCarloMPI
	rm -f MonteCarloBench
