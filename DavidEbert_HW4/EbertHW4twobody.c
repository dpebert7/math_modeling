/*
David Ebert
Homework 4 - Two Body on CPU

Output:
2 bodies swirl around for awhile.
Seems right.

*/
	

// gcc EbertHW4twobody.c -o temp3D -lglut -lm -lGLU -lGL
// ./temp3D
// To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.141592654

#define N 2  // 2 bodies

#define XWindowSize 700 
#define YWindowSize 700

#define STOP_TIME 100.0  
#define DT        0.0001 //Step size, global variable

#define GRAVITY 1.0
#define MASSBODY1 10.0
#define MASSBODY2 10.0

#define DRAW 10 //Not sure?

// Globals  //2-vectors for storing position & mass in x, y, and z directions.
double px[N], py[N], pz[N], vx[N], vy[N], vz[N], fx[N], fy[N], fz[N], mass[N], G; 

void set_initail_conditions() // Initial conditions
{
	G = GRAVITY;
	mass[0] = MASSBODY1;
	mass[1] = MASSBODY2;
	
	px[0] = 0.5; // yellow ball's starting position
	py[0] = 0.0;
	pz[0] = 0.0;
	
	px[1] = -0.5; // purple ball's starting position
	py[1] = 0.0;
	pz[1] = 0.0;
	
	vx[0] = 0.0; // yellow ball's starting velocity
	vy[0] = 1.0;
	vz[0] = 0.0;
	
	vx[1] = 0.0; // purple ball's starting velocity
	vy[1] = -1.0;
	vz[1] = 0.0;
}

void draw_picture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	glColor3d(1.0,1.0,0.5);
	glPushMatrix();
	glTranslatef(px[0], py[0], pz[0]);
	glutSolidSphere(0.1,20,20);
	glPopMatrix();
	
	glColor3d(1.0,0.5,1.0);
	glPushMatrix();
	glTranslatef(px[1], py[1], pz[1]);
	glutSolidSphere(0.1,20,20);
	glPopMatrix();
	
	glutSwapBuffers();
}

int n_body()
{
	double fx[N], fy[N], fz[N], f; 
	double dx,dy,dz,d, dt;
	double dvx,dvy,dvz,close_seperate;
	int    tdraw = 0; int   tprint = 0;
	double  time = 0.0;
	int i,j;
	
	dt = DT;

	while(time < STOP_TIME)
	{
		for(i=0; i<N; i++)
		{
			fx[i] = 0.0;
			fy[i] = 0.0;
			fz[i] = 0.0;
		}

		//Get forces
		for(i=0; i<N; i++)
		{
			for(j=i+1; j<N; j++)
			{
				dx = px[i]-px[j]; // Find distance between bodies
				dy = py[i]-py[j];
				dz = pz[i]-pz[j];
				d = sqrt(dx*dx + dy*dy + dz*dz);

				fx[j] = G*mass[i]*(px[i]-px[j])/(d*d*d); // Forces
				fy[j] = G*mass[i]*(py[i]-py[j])/(d*d*d);
				fz[j] = G*mass[i]*(pz[i]-pz[j])/(d*d*d);

				fx[i] = G*mass[i]*(px[j]-px[i])/(d*d*d); // Forces
				fy[i] = G*mass[i]*(py[j]-py[i])/(d*d*d);
				fz[i] = G*mass[i]*(pz[j]-pz[i])/(d*d*d);
			}
		}

		//Move elements
		for(i=0; i<N; i++)
		{			
			px[i]=px[i]+vx[i]*dt; //update position
			py[i]=py[i]+vy[i]*dt;
			pz[i]=pz[i]+vz[i]*dt;

			vx[i] = vx[i] + fx[i]*dt; // update velocities
			vy[i] = vy[i] + fy[i]*dt;
			vz[i] = vz[i] + fz[i]*dt;
		}

		if(tdraw == DRAW) 
		{
			draw_picture();
			tdraw = 0;
		}

		time += dt;
		tdraw++;
		tprint++;
	}
}

void control()
{	
	int    tdraw = 0;
	double  time = 0.0;

	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);

	set_initail_conditions();
	
	draw_picture();
	
	n_body();
	
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	control();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 150.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}
