/*
David Ebert
Homework 2 - Part 1 - GPU Addition
*/

/*
For assignment 2 part 1, do the following:
	-Take the GPU vector addition program from HW1 and make it work with multiple blocks.
	-You will enter a number N which will be given to you to setup only just enough blocks to work with.
	-Add the vectors as normal.

The code below works for N values up to at least 10 million.

Printed results for N = 10 000:
Time in milliseconds= 0.090000000000000
Last Values are A[9999] = 9999.000000000000000  B[9999] = 9999.000000000000000  C[9999] = 19998.000000000000000\
*/


// To compile and run: nvcc EbertHW2part1.cu -O3 -o temp -lcudart -run
// To run: ./temp
#include <sys/time.h>
#include <stdio.h>

//Length of vectors to be added.
#define N 10000

float *A_CPU, *B_CPU, *C_CPU; //CPU pointers

float *A_GPU, *B_GPU, *C_GPU; //GPU pointers

dim3 dimBlock; //This variable will hold the Dimensions of your block

void AllocateMemory()
{					
	//Allocate Device (GPU) Memory, & allocates the value of the specific pointer/array
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaMalloc(&C_GPU,N*sizeof(float));

	//Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));

}

//Loads values into vectors that we will add.
void Innitialize()
{
	int i;
	
	for(i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)i;
	}
}

//Cleaning up memory after we are finished.
void CleanUp(float *A_CPU,float *B_CPU,float *C_CPU,float *A_GPU,float *B_GPU,float *C_GPU)  //free
{
	free(A_CPU); free(B_CPU); free(C_CPU);
	cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);
}

//This is the kernel. It is the function that will run on the GPU.
//It adds vectors A and B then stores result in vector C
__global__ void Addition(float *A, float *B, float *C, int n)
{
	int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	if(thread_id< n){
		C[thread_id] = A[thread_id] + B[thread_id];
	}

}

int main()
{
	int i;
	timeval start, end;
	//cudaError_t err; // Not sure what this is. Turning it off.
	
	//Set the thread structure that you will be using on the GPU	
	//SetUpCudaDevices(); // Not sure what this function is. Turning it off.

	//Partitioning off the memory that you will be using.
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();
	
	//Starting the timer
	gettimeofday(&start, NULL);

	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	
	//Calling the Kernel (GPU) function.	
	Addition<<<(N+1023)/1024,1024>>>(A_GPU, B_GPU, C_GPU, N);
	
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);

	//Stopping the timer
	gettimeofday(&end, NULL);

	//Calculating the total time used in the addition and converting it to milliseconds.
	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	
	//Displaying the time 
	printf("Time in milliseconds= %.15f\n", (time/1000.0));	

	// Displaying vector info you will want to comment out the vector print line when your
	//vector becomes big. This is just to make sure everything is running correctly.	
	for(i = 0; i < N; i++)		
	{		
		//printf("A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", i, A_CPU[i], i, B_CPU[i], i, C_CPU[i]);
	}

	//Displaying the last value of the addition for a check when all vector display has been commented out.
	printf("Last Values are A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", N-1, A_CPU[N-1], N-1, B_CPU[N-1], N-1, C_CPU[N-1]);
	
	//You're done so cleanup your mess.
	CleanUp(A_CPU,B_CPU,C_CPU,A_GPU,B_GPU,C_GPU);	
	
	return(0);
}
