/*For HW3:
-Find the mistake in the provided code and fix it.
-Understand how the code works well enough to go line-by-line.

Mitake is in the second while loop of the display function
(line 70), where the while loop uses <= instead of <.

When "<=" is used, there is one more pixel per row than there
are columns, causing each subsequent row to shift 
an additional 1 pixel right.
*/

//nvcc EbertHW3fractal.cu -o temp -lglut -lGL -lm

#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


#define A  -0.624 // starting parameter's real part
#define B  0.4351 // starting parameter's imaginary part

unsigned int window_width = 1024; // number of pixels to use, width
unsigned int window_height = 1024;

float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width);
float stepSizeY = (yMax - yMin)/((float)window_height);

float color (float x, float y) // function for determining a pixel's color
{
	float mag,maxMag,t1;
	float maxCount = 200;
	float count = 0;
	maxMag = 10;
	mag = 0.0;
	
	while (mag < maxMag && count < maxCount) 
	{
		t1 = x;			// store x value as t1 for later
		x = x*x - y*y + A;	// real part of f(z) = z^2 + c
		y = (2.0 * t1 * y) + B; // imaginary part of f(z) = z^2+c
		mag = sqrt(x*x + y*y);	// check magnitude. If it's above 10, then break.
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

void display(void) 
{ 
	float *pixels; 
	float x, y;
	int k;

	pixels = (float *)malloc(window_width*window_height*3*sizeof(float));
	k=0;

	y = yMin;
	
	while(y < yMax)  // remove "="!
	{
		x = xMin;
		//printf("%f", y); // Check to make sure y is incrementing correctly.
		while(x < xMax)	// remove "="! This is the main cluprit!
				// When "<=" is used, there is one more pixel per row than there
				// are columns, causing each subsequent row to shift 
				// an additional 1 pixel right.
		{	
			
			pixels[k] = 0.0;		//Red off
			pixels[k+1] = color(x,y); 	//Green on or off returned from color
			pixels[k+2] = color(x,y); 	//Blue on or off returned from color
			k=k+3;				//Skip to next pixel
			//printf("%f", x); // Check to make sure x is incrementing correctly.
			x += stepSizeX;
		}
		y += stepSizeY;
	}

	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
}

int main(int argc, char** argv) //call functions here
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
   	glutCreateWindow("Fractals man, fractals.");
   	glutDisplayFunc(display);
   	glutMainLoop();
}

