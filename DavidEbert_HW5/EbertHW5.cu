/*
David Ebert
Homework 5 - Julia Set on GPU 

Output:
A nice fractal!
*/


//nvcc EbertHW5.cu -o temp -lglut -lGL -lm

#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


#define A  -0.624 // starting value's real part
#define B  0.4351 // starting value's imaginary part

unsigned int window_width = 1024; // number of pixels to use, width
unsigned int window_height = 1024;

float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width);
float stepSizeY = (yMax - yMin)/((float)window_height);

float *PIXELS_CPU;
float *PIXELS_GPU;

void AllocateMemory()
{				
	//Allocate Device (GPU) Memory, & allocates the value of the specific pointer/array
	cudaMalloc(&PIXELS_GPU,window_width*window_height*3*sizeof(float));
	
	//Allocate Host (CPU) Memory
	PIXELS_CPU = (float *)malloc(window_width*window_height*3*sizeof(float));
}



__device__ float color (float x, float y) // function for determining a pixel's color
{
	float mag,maxMag,t1;
	float maxCount = 200;
	float count = 0;
	maxMag = 10;
	mag = 0.0;
	
	while (mag < maxMag && count < maxCount) 
	{
		t1 = x;			
		x = x*x - y*y + A;	
		y = (2.0 * t1 * y) + B; 
		mag = sqrt(x*x + y*y);	
		count++;
	}
	if(count < maxCount) 
	{
		return(1.0);
	}
	else
	{
		return(0.0);
	}
}



__global__ void Kernel(float *pixels)
{ 
	// Very hard-coded, but it works!
	int thread_id = threadIdx.x + blockIdx.x*blockDim.x;
	pixels[3*thread_id]   = color(-2.0 + 4*threadIdx.x/1024.0, -2.0 + 4*blockIdx.x/1024.0);
	pixels[3*thread_id+1] = color(-2.0 + 4*threadIdx.x/1024.0, -2.0 + 4*blockIdx.x/1024.0);
	pixels[3*thread_id+2] = color(-2.0 + 4*threadIdx.x/1024.0, -2.0 + 4*blockIdx.x/1024.0);
}



void display(void) 
{ 
	Kernel<<<1024,1024>>>(PIXELS_GPU);
	
	//Copy Memory from GPU to CPU (Is this necessary?)
	cudaMemcpyAsync(PIXELS_CPU, PIXELS_GPU, window_width*window_height*3*sizeof(float), cudaMemcpyDeviceToHost);
	
	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, PIXELS_CPU); 
	glFlush(); 
	
}



int main(int argc, char** argv) //call functions here
{ 
	printf("Hello World \n");
	
	AllocateMemory();
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
   	glutCreateWindow("This is David's GPU fractal, hopefully.");
   	glutDisplayFunc(display); // This function calls the kernel function.
   	
   	printf("Last Value is PIXELS_CPU[%d] = %.15f \n", 1024*1024*3, PIXELS_CPU[1024*1024*3]);
   	printf("Hello again \n");
   	
   	glutMainLoop();
}
