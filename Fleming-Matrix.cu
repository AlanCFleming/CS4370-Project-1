#include <stdio.h>
#include <cuda.h>
#define SIZE 100 //N can not be larger than 256


void add_matrix_cpu(int *a, int *b, int *c, int N){
	int i, j index;
	for(i=0; i<N; i++){
		for(j = 0; j<N; j++){
			index = i*N+j;
			c[index] = a[index] + b[index];
		}
	}
}
