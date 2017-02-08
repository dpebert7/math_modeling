/*
David Ebert
Homework 6 - GPU Dot Product

HW 6 Assignment:
	-Dot product on GPU
	-Add as much up on the GPU as possible
	-Final addition is alright on the CPU

Output for N = 5000:
Vector A is 0.5 repeated 5000 times.
Vector B is 2.0 repeated 5000 times.
The dot product of A and B should be equal to N = 5000.

Output for N = 5000 (Seems to work up to at least 50 000):

Time in milliseconds= 0.318000000000000
A[0] = 2.00000  B[0] = 0.50000  C[0] = 1024.00000
A[1024] = 2.00000  B[1024] = 0.50000  C[1024] = 1024.00000
A[2048] = 2.00000  B[2048] = 0.50000  C[2048] = 1024.00000
A[3072] = 2.00000  B[3072] = 0.50000  C[3072] = 1024.00000
A[4096] = 2.00000  B[4096] = 0.50000  C[4096] = 904.00000
A and B are vectors of length 5000
The number of threads per block is 1024 (a power of 2) 
The number of blocks is 5
The total number of threads is 5120
Finally, the dot product of A and B is... 5000 	<-- This is correct.

*/


// To compile and run: nvcc EbertHW6.cu -O3 -o temp -lcudart -run
// To run: ./temp
#include <sys/time.h>
#include <stdio.h>

//Length of vectors to be added.
#define N 5000		// Length of A and B vectors. Works for values up to 10000
#define numThreads 1024	// This is really the number of threads per block. Should be power of 2.
			// For some reason this isn't working with values greater than 64 if N is big. 
			// I think it's a sync thread problem.
#define M ((N+numThreads-1)/numThreads)*numThreads // M is the total number of threds, just larger than N


float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 dimBlock; //This variable will hold the Dimensions of your block

void AllocateMemory()
{					
	//Allocate Device (GPU) Memory, & allocates the value of the specific pointer/array
	cudaMalloc(&A_GPU,M*sizeof(float));
	cudaMalloc(&B_GPU,M*sizeof(float));
	cudaMalloc(&C_GPU,M*sizeof(float));

	//Allocate Host (CPU) Memory
	A_CPU = (float*)calloc(M,sizeof(float));
	B_CPU = (float*)calloc(M,sizeof(float));
	C_CPU = (float*)calloc(M,sizeof(float));

}

//Loads values into vectors that we will add.
void Innitialize()
{
	int i;

	for(i = 0; i < N; i++)
	{
		A_CPU[i] = 2.0;	
		B_CPU[i] = 0.5;	// dot product should be N
		//_CPU[i] = (float)i;	
		//B_CPU[i] = (float)i;
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
__global__ void Dot(float *A, float *B, float *C)
{
	int thread_id; //threadIdx.x + blockIdx.x * blockDim.x;
	for(thread_id = threadIdx.x + blockIdx.x * blockDim.x; thread_id<N; thread_id += (blockDim.x * gridDim.x)){
		C[thread_id] = A[thread_id]*B[thread_id];
	}
	
	__syncthreads();
	
	//int theThread = threadIdx.x;
	int i = numThreads/2;
	while(i!=0){
		for(thread_id = threadIdx.x + blockIdx.x * blockDim.x; thread_id<N; thread_id += (blockDim.x*gridDim.x)){
		if(threadIdx.x <i){
			C[thread_id] += C[thread_id+i];
			}
		__syncthreads();
		}
	i/=2;
	}
}

int main()
{
	//printf("Hello World \n");
	// print number of threads (should be multiple of numThreads)
	
	
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
	cudaMemcpyAsync(A_GPU, A_CPU, M*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_GPU, B_CPU, M*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(C_GPU, C_CPU, M*sizeof(float), cudaMemcpyHostToDevice);
	
	//Calling the Kernel (GPU) function.	
	Dot<<<M,numThreads>>>(A_GPU, B_GPU, C_GPU);
	
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, M*sizeof(float), cudaMemcpyDeviceToHost);

	//Stopping the timer
	gettimeofday(&end, NULL);

	//Calculating the total time used in the addition and converting it to milliseconds.
	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	
	//Displaying the time 
	printf("Time in milliseconds= %.15f\n", (time/1000.0));	

	// Displaying vector info you will want to comment out the vector print line when your
	//vector becomes big. This is just to make sure everything is running correctly.	
	for(i = 0; i < M; i++)		
	{		
		//printf("A[%d] = %.5f  B[%d] = %.5f  C[%d] = %.5f\n", i, A_CPU[i], i, B_CPU[i], i, C_CPU[i]);
	}
	
	// calculate and print final sum
	int sum;
	int j;
	for(j=0;j<N;j=j+numThreads){
		sum+=C_CPU[j];
		printf("A[%d] = %.5f  B[%d] = %.5f  C[%d] = %.5f\n", j, A_CPU[j], j, B_CPU[j], j, C_CPU[j]);
	}
	printf("A and B are vectors of length %d\n", N);
	printf("The number of threads per block is %d (a power of 2) ", numThreads);
	printf("\nThe number of blocks is %d\n", M/numThreads);
	printf("The total number of threads is %d\n", M);
	printf("Finally, the dot product of A and B is... %d\n", sum);

	//You're done so cleanup your mess.
	CleanUp(A_CPU,B_CPU,C_CPU,A_GPU,B_GPU,C_GPU);	
	
	return(0);
}
