#include <stdio.h>
#include <cuda.h>
//Code writen by Alan Fleming

void add_matrix_cpu(int *a, int *b, int *c, int N){
	int i, j, index;
	for( i = 0; i<N; i++){
		for( j = 0; j<N; j++){
			index = i*N+j;
			c[index] = a[index] + b[index];
		}
	}
}

__global__ void add_matrix_gpu(int *a, int *b, int *c, int N){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int index = row * N + col;
	if( row < N && col < N) {
		c[index] = a[index] + b[index];
	}
}

void printMatrix(int *m, int N){
	for( int i = 0; i < N; i++){
		for( int j = 0; j < N; j++){
			printf("%d ", m[i * N + j]);
		}
		printf("\n");
	}
}


int verifyMatrix(int *a, int *b, int N){
	for( int i = 0; i < N; i++){
		for( int j = 0; j < N; j++){
			if(a[i * N + j] != b[i * N + j]){
				printf("TEST FAILED\n");
				return 1;
			}
		}
	}
	printf("TEST PASSED\n");
	return 0;
}

int main(int argc, char *argv[]){

	//assign matrix and block size
	const int MATRIXSIZE = argv[1];
	const int BLOCKSIZE = argv[2];

	//allocate system memory for array
	int *a = (int *)malloc(sizeof(int) * MATRIXSIZE * MATRIXSIZE );	//first matrix
	int *b = (int *)malloc(sizeof(int) * MATRIXSIZE * MATRIXSIZE ); //second matrix
	int *c = (int *)malloc(sizeof(int) * MATRIXSIZE * MATRIXSIZE ); //result from CPU
	int *d = (int *)malloc(sizeof(int) * MATRIXSIZE * MATRIXSIZE ); //result from gpu

	//initialize a and b for addition
	int init = 1325;
	for( int i = 0; i < MATRIXSIZE; i++){
		for( int j = 0; j < MATRIXSIZE; j++){
			init = 3125 * init % 65536;
			a[ i * MATRIXSIZE + j ] = (init - 32768)/6553;
			b[ i * MATRIXSIZE + j ] = init % 1000;
		}
	}

	//print initial matrix a and b
	printf("a \n --------------------- \n");
	printMatrix(a, MATRIXSIZE);

	printf("b \n --------------------- \n");
	printMatrix(b, MATRIXSIZE);

	//add matrix using cpu
	add_matrix_cpu(a, b, c, MATRIXSIZE);
	
	//print the result
	printf("c \n --------------------- \n");
	printMatrix(c, MATRIXSIZE);

	//allocate memory on device
	int *dev_a, *dev_b, *dev_c;

	cudaMalloc((void **)(&dev_a),MATRIXSIZE * MATRIXSIZE * sizeof(int));
	cudaMalloc((void **)(&dev_b),MATRIXSIZE * MATRIXSIZE * sizeof(int));
	cudaMalloc((void **)(&dev_c),MATRIXSIZE * MATRIXSIZE * sizeof(int));

	//copy memory to device
	cudaMemcpy(dev_a,a, MATRIXSIZE * MATRIXSIZE * sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b, MATRIXSIZE * MATRIXSIZE * sizeof(int),cudaMemcpyHostToDevice);
	
	//calculate gridWidth
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);

	int gridWidth = ceil((MATRIXSIZE-1)/double(dimBlock.x));

	//define dimGrid
	dim3 dimGrid(gridWidth, gridWidth,1);

	//add matrix using gpu
	add_matrix_gpu<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, MATRIXSIZE);

	//copy memory from device
	cudaMemcpy(d,dev_c, MATRIXSIZE * MATRIXSIZE * sizeof(int),cudaMemcpyDeviceToHost);

	//print the result
	printf("d \n --------------------- \n");
	printMatrix(d, MATRIXSIZE);

	//verify the results
	verifyMatrix(c, d, MATRIXSIZE);

	//free memory
	free(a);
        free(b);
        free(c);
        free(d); 
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
	//exit program
	return 0;
}
