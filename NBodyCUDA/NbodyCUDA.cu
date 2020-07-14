#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include "NBody.h"
#include "NBodyVisualiser.h"
#include <stdbool.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define USER_NAME "Mengwen Han"		

void print_help();
int D;
int N;
int stepNum = 0;


__constant__ int N_gpu; //Global variables - number of stars
__constant__ int D_gpu;

bool I = true;//Use visualization by default
bool F = false;//Default non file read in
float* ax, * ay; //Global variable - Star acceleration array
nbody* planet; //All planets
nbody_soa* planet_soa; //All planets
float* heat;//heat map

nbody* dplanet; //All planets
nbody_soa* dplanet_soa; //All planets

float* dheat;//heat map
float* dfx, * dfy;
float* dfmatx, * dfmaty;
__constant__ float s2 = 4.0f;

/*Initializing a small planet*/
void nbodyInit(int i, float xxPos, float yyPos, float xxVel, float yyVel, float mass) {
	planet_soa->x[i] = xxPos;
	planet_soa->y[i] = yyPos;
	planet_soa->vx[i] = xxVel;
	planet_soa->vy[i] = yyVel;
	planet_soa->m[i] = mass;
}


/*Calculate the square of the distance between two planets*/
double calcDistance(nbody a, nbody b) {
	return   pow(((double)a.x - (double)b.x), 2) + pow(((double)a.y - (double)b.y), 2);
}

/*In order to calculate the x-direction gravity between two stars, the E parameter is introduced*/
float calcForceX(nbody a, nbody b) {

	float fx = (b.m * (b.x - a.x)) / ((float)pow(calcDistance(a, b) + (float)pow(SOFTENING, 2), 1.5));
	return fx;
}
/*In order to calculate the x-direction gravity between two stars, the E parameter is introduced*/
float calcForceY(nbody a, nbody b) {
	//float fy = (G * a.m * b.m * (b.y - a.y)) / ((float)pow(calcDistance(a, b) + (float)pow(SOFTENING, 2), 1.5));
	float fy = (b.m * (b.y - a.y)) / ((float)pow(calcDistance(a, b) + (float)pow(SOFTENING, 2), 1.5));
	return fy;
}

/*Calculate the static gravity at the current time, X direction*/
float calcNetForceX(nbody* a, nbody* all, int N) {
	float netFx = 0;
	for (int i = 0; i < N; i++) {

		netFx += calcForceX(*a, all[i]);

	}
	netFx = netFx * G * (float)(a->m);
	return netFx;
}

/*Calculate the static gravity at the current time, Y direction*/
float calcNetForceY(nbody* a, nbody* all, int N) {
	float netFy = 0;
	for (int i = 0; i < N; i++) {

		netFy += calcForceY(*a, all[i]);
	}
	netFy = netFy * G * (float)(a->m);
	return netFy;
}

/*Update planet parameters by acceleration in time period*/
void updatePlanet(nbody* a, float ax, float ay) {
	a->vx += ax * dt;
	a->vy += ay * dt;
	a->x += a->vx * dt;
	a->y += a->vy * dt;


	//The following condition is that the planet bounces back when it collides with the boundary, 
	//and it should be replaced by collision and annihilation when it is formally simulated
	if (a->x >= 1 || a->y >= 1 || a->x <= 0 || a->y <= 0) {
		a->vx = -a->vx;
		a->vy = -a->vy;
	}


}


/*Generate a random planet*/
void createOnePlanet(nbody_soa* p_soa, int i) {

	p_soa->x[i] = 1.0 * rand() / RAND_MAX;
	p_soa->y[i] = 1.0 * rand() / RAND_MAX;
	p_soa->vx[i] = 0;
	p_soa->vy[i] = 0;
	p_soa->m[i] = 1.0 / N;

}
void createOnePlanet2(nbody* p) {

	p->x = 1.0 * rand() / RAND_MAX;
	p->y = 1.0 * rand() / RAND_MAX;
	p->vx = 0;
	p->vy = 0;
	p->m = 1.0 / N;
}


/*Initial heat map*/
void heatInit() {
	for (int i = 0; i < D * D; i++) {
		heat[i] = 0;
	}
}



//Process the read row of data. If the data is incomplete, it will be supplemented randomly
void readOnePlanet(char str[], float* planetParameters)
{
	int delims[6] = { -1 };
	char num[5][20] = { 0 };
	int flag = 1;
	int temp = 0;

	for (int i = 0; i <= strlen(str); i++) {
		if (str[i] == ',') {
			delims[flag] = i;
			flag++;
		}
	}
	delims[5] = strlen(str);
	for (int i = 0; i < 5; i++) {
		for (int j = delims[i] + 1; j < delims[i + 1]; j++) {
			if (str[j] != ' ' && str[j] != 'f') {
				num[i][temp] = str[j];
				temp++;
			}
		}
		temp = 0;
	}

	for (int i = 0; i < 2; i++) {
		if (strlen(num[i]) == 1) planetParameters[i] = (float)rand() / (RAND_MAX + 1.0);
		else planetParameters[i] = atof(num[i]);
	}

	for (int i = 2; i < 4; i++) {
		if (strlen(num[i]) == 1) planetParameters[i] = 0;
		else planetParameters[i] = atof(num[i]);
	}

	if (strlen(num[4]) == 1) {
		planetParameters[4] = 1.0 / N;

	}
	else planetParameters[4] = atof(num[4]);


}

//Read all planet data from the file and save it in planet
int readPlanet(char* input_file)
{
	int a = 0;
	FILE* fp;
	char strLine[1024];								//Read buffer
	int flag = 0;
	float planetParameters[5] = { 0 };
	if ((fp = fopen(input_file, "r")) == NULL)		//Judge whether the file exists and is readable
	{
		printf("Open Falied!");
		return -1;
	}
	while (!feof(fp))									//Loop through each line until the end of the file
	{
		if (fgets(strLine, 1024, fp) != NULL) {					    //Read the line contents of the file pointed to by fp to the strline buffer
			if (strLine[0] != '#' && (strLine[0] != '\n')) {
				readOnePlanet(strLine, planetParameters);
				nbodyInit(flag, planetParameters[0], planetParameters[1], planetParameters[2], planetParameters[3], planetParameters[4]);
				flag++;

			}
		}
	}

	if (flag == N) {
		a = 1;
	}
	fclose(fp);
	return a;
}

//No visual one-step update of all planets - CPU
void stepICPU(int stepNum) {
	for (int j = 0; j < stepNum; j++) {
		for (int i = 0; i < N; i++)
		{
			updatePlanet(&planet[i], calcNetForceX(&planet[i], planet, N) / planet[i].m, calcNetForceY(&planet[i], planet, N) / planet[i].m);
		}
	}
}


//No visual single step update of all planets - OpenMP
void stepIOPENMP(int stepNum) {
	int j;
#pragma omp parallel for
	for (j = 0; j < stepNum; j++)
	{
		int i;
#pragma omp parallel for
		for (i = 0; i < N; i++)
		{
			updatePlanet(&planet[i], calcNetForceX(&planet[i], planet, N) / planet[i].m, calcNetForceY(&planet[i], planet, N) / planet[i].m);
		}
	}
}



// Single step update of all planets with CPU
void stepCpu(void)
{
	//TODO: Perform the main simulation of the NBody system
		/*After a time slice DT, the parameters of all the stars have been updated*/
	heatInit();
	int clockstart, clockend;
	clockstart = clock();
	int i;
	for (i = 0; i < N; i++)
	{
		updatePlanet(&planet[i], calcNetForceX(&planet[i], planet, N) / planet[i].m, calcNetForceY(&planet[i], planet, N) / planet[i].m);

		heat[(int)(planet[i].y * D) * D + (int)(planet[i].x * D)] ++;

	}
	clockend = clock();

	for (int i = 0; i < D * D; i++)
		heat[i] /= (N / 3);
}

// Single step update of all planets with OpenMP
void stepOmp(void)
{
	//TODO: Perform the main simulation of the NBody system
		/*After a time slice DT, the parameters of all the stars have been updated*/
	heatInit();
#pragma omp parallel
	for (int i = 0; i < N; i++)
	{
		updatePlanet(&planet[i], calcNetForceX(&planet[i], planet, N) / planet[i].m, calcNetForceY(&planet[i], planet, N) / planet[i].m);
		heat[(int)(planet[i].y * D) * D + (int)(planet[i].x * D)] ++;
	}
	for (int i = 0; i < D * D; i++)
		heat[i] /= (N / 3);
}

// copy data to gpu
void initCuda(void)
{

	//dplanet; //All planets
	// dheat;//heat map


	cudaMalloc((void**)&dfx, N * sizeof(float));

	cudaMalloc((void**)&dfy, N * sizeof(float));

	cudaMalloc((void**)&dfmatx, N * N * sizeof(float));

	cudaMalloc((void**)&dfmaty, N * N * sizeof(float));

	// cudaMalloc dplanet_soa
	dplanet_soa = (nbody_soa*)malloc(sizeof(nbody_soa));

	cudaMalloc((void**)&dplanet_soa->x, N * sizeof(float));

	cudaMalloc((void**)&dplanet_soa->y, N * sizeof(float));

	cudaMalloc((void**)&dplanet_soa->vx, N * sizeof(float));

	cudaMalloc((void**)&dplanet_soa->vy, N * sizeof(float));

	cudaMalloc((void**)&dplanet_soa->m, N * sizeof(float));


	cudaMalloc((void**)&dheat, D * D * sizeof(float));

	// cudaMemcpy

	cudaMemcpy(dplanet_soa->x, planet_soa->x, N * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(dplanet_soa->y, planet_soa->y, N * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(dplanet_soa->vx, planet_soa->vx, N * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(dplanet_soa->vy, planet_soa->vy, N * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(dplanet_soa->m, planet_soa->m, N * sizeof(float), cudaMemcpyHostToDevice);

	// heat 
	cudaMemcpy(&dheat[0], &heat[0], D * D * sizeof(float), cudaMemcpyHostToDevice);











	// cudaMemcpy
	cudaMemcpyToSymbol(N_gpu, &N, sizeof(int));
	cudaMemcpyToSymbol(D_gpu, &D, sizeof(int));



}


//      N*N   Force  Parallel Implementation
__global__ void computeForcePair(float* fmatx, float* fmaty, float* x, float* y, float* m, float* fx, float* fy)
{
	float distance;
	float coef;
	float dx;
	float dy;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N_gpu * N_gpu)
	{
		return;
	}
	int i = tid / N_gpu;
	int j = tid % N_gpu;

	if (i == j)
	{
		fmatx[tid] = 0;
		fmaty[tid] = 0;
		return;
	}


	dx = x[j] - x[i];
	dy = y[j] - y[i];
	distance = dx * dx + dy * dy;
	coef = G * m[j] / powf(distance + s2, 1.5);
	fmatx[tid] = coef * dx;
	fmaty[tid] = coef * dy;

	atomicAdd(&fx[i], fmatx[tid]);
	atomicAdd(&fy[i], fmaty[tid]);

	return;
}




//update all planet
__global__ void update(float* fx, float* fy, float* x, float* y, float* vx, float* vy)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N_gpu)
	{
		return;
	}
	//vx[tid] += dt * (fx[tid]/m[tid]);
	//vy[tid] += dt * (fy[tid]/m[tid]);
	vx[tid] += dt * fx[tid];
	vy[tid] += dt * fy[tid];

	x[tid] += dt * vx[tid];
	y[tid] += dt * vy[tid];
}

//comute heat
__global__ void computeHeat(float* x, float* y, float* h)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N_gpu)
	{
		return;
	}

	int x_index = (int)(x[tid] / (1.0 / D_gpu));
	int y_index = (int)(y[tid] / (1.0 / D_gpu));

	if (x_index < 0 || x_index >= D_gpu || y_index < 0 || y_index >= D_gpu) {
		return;
	}

	//comute heat
	atomicAdd(&h[y_index * D_gpu + x_index], 1.0f / N_gpu * D_gpu);
}


void stepCuda(void)
{
	dim3 Block = { 256 };
	dim3 Grid = { (unsigned)(N / 256) + 1 };

	//cudaMemset
	cudaMemset(dheat, 0, D * D * sizeof(float));
	cudaMemset(dfmatx, 0, N * N * sizeof(float));
	cudaMemset(dfmatx, 0, N * N * sizeof(float));
	cudaMemset(dfx, 0, N * sizeof(float));
	cudaMemset(dfy, 0, N * sizeof(float));

	// all force
	computeForcePair << <N * N / 256 + 1, 256 >> > (dfmatx, dfmaty, dplanet_soa->x, dplanet_soa->y, dplanet_soa->m, dfx, dfy);

	// update 
	update << <Grid, Block >> > (dfx, dfy, dplanet_soa->x, dplanet_soa->y, dplanet_soa->vx, dplanet_soa->vy);

	//heat
	computeHeat << <Grid, Block >> > (dplanet_soa->x, dplanet_soa->y, dheat);


}



void stepICUDA(int stepNum)
{

	dim3 Block = { 256 };
	dim3 Grid = { (unsigned)(N_gpu / 256) + 1 };

	initCuda();
	
	for (int i = 0; i < stepNum; i++)
	{
		//cudaMemset
		cudaMemset(dheat, 0, D * D * sizeof(float));
		cudaMemset(dfmatx, 0, N * N * sizeof(float));
		cudaMemset(dfmatx, 0, N * N * sizeof(float));
		cudaMemset(dfx, 0, N * sizeof(float));
		cudaMemset(dfy, 0, N * sizeof(float));
		
		// all force
		computeForcePair << <N * N / 256 + 1, 256 >> > (dfmatx, dfmaty, dplanet_soa->x, dplanet_soa->y, dplanet_soa->m, dfx, dfy);

		// update 
		update << <Grid, Block >> > (dfx, dfy, dplanet_soa->x, dplanet_soa->y, dplanet_soa->vx, dplanet_soa->vy);

		//heat
		computeHeat << <Grid, Block >> > (dplanet_soa->x, dplanet_soa->y, dheat);

	}

	cudaFree(&dplanet_soa->x[0]);
	cudaFree(&dplanet_soa->y[0]);
	cudaFree(&dplanet_soa->vx[0]);
	cudaFree(&dplanet_soa->vy[0]);
	cudaFree(&dplanet_soa->m[0]);
	cudaFree(&dheat[0]);
	cudaFree(&dfx[0]);
	cudaFree(&dfy[0]);




	return;
}


int main(int argc, char* argv[]) {


	char* filePath;

	//DONE: Processes the command line arguments
		//argc in the count of the command arguments
		//argv is an array (of length argc) of the arguments. The first argument is always the executable name (including path)


	// N
	char* num = argv[1];
	for (int i = 0; num[i]; i++)
	{
		if (num[i] > '9' || num[i] < '0')
		{
			printf("N is not int\n");
			return 0;
		}
	}

	N = atoi(argv[1]);
	if ((N <= 0)) {
		printf("N  is wrong");
		exit(0);
	}
	//   D
	char* num2 = argv[2];
	for (int i = 0; num2[i]; i++)
	{
		if (num2[i] > '9' || num2[i] < '0')
		{
			printf("D is not int\n");
			return 0;
		}
	}

	D = atoi(argv[2]);
	if ((D <= 0)) {
		printf("D  is wrong");
		exit(0);
	}

	// I
	for (int i = 4; i < argc; i++) {
		if (!strcmp(argv[i], "-i")) {
			I = false;
			char* num3 = argv[i + 1];
			for (int i = 0; num3[i]; i++)
			{
				if (num3[i] > '9' || num3[i] < '0')
				{
					printf("I is not int\n");
					return 0;
				}
			}
			stepNum = atoi(argv[i + 1]);

		}

		// f
		if (!strcmp(argv[i], "-f")) {
			filePath = argv[i + 1];
			F = true;
		}
	}


	//DONE: Allocate any heap memory
	ax = (float*)malloc(N * sizeof(float));
	ay = (float*)malloc(N * sizeof(float));
	planet_soa = (nbody_soa*)malloc(N * sizeof(nbody_soa));
	planet_soa->x = (float*)malloc(N * sizeof(float));
	planet_soa->y = (float*)malloc(N * sizeof(float));
	planet_soa->vx = (float*)malloc(N * sizeof(float));
	planet_soa->vy = (float*)malloc(N * sizeof(float));
	planet_soa->m = (float*)malloc(N * sizeof(float));
	heat = (float*)malloc((D * D) * sizeof(float));
	planet = (nbody*)malloc(N * sizeof(nbody));




	//DONE: Depending on program arguments, either read initial data from file or generate random data.
	if (F == true) {
		//When reading the planet from the file, 
		//there is no check for out of range or insufficient.
		//Because the topic convention n is the same as when reading it,
		//the runtime error will be automatically thrown if it is different
		int a = readPlanet(filePath);
		if (a == 0) {
			printf("RuntimeError:N is wrong");
			exit(0);
		}
	}
	else {
		/*Randomly generated n small planets*/
		for (int i = 0; i < N; i++) {
			createOnePlanet(planet_soa, i);
			createOnePlanet2(&planet[i]);

		}
	}

	//DONE: Depending on program arguments, either configure and start the visualiser or perform a fixed number of simulation steps (then output the timing results).
	//Determine the read operation mode CPU or OpenMP
	if (I == true) {

		if (!strcmp(argv[3], "CPU"))
		{
			initViewer(N, D, CPU, stepCpu);
			setNBodyPositions(planet);
			setHistogramData(heat);
			startVisualisationLoop();
		}
		else if (!strcmp(argv[3], "OPENMP"))
		{
			initViewer(N, D, OPENMP, stepOmp);
			setNBodyPositions(planet);
			setHistogramData(heat);
			startVisualisationLoop();
		}
		else
		{

			initCuda();
			initViewer(N, D, CUDA, stepCuda);

			setNBodyPositions2f(dplanet_soa->x, dplanet_soa->y);
			setHistogramData(dheat);
			startVisualisationLoop();
			// free cuda
			cudaFree(&dplanet_soa->x[0]);
			cudaFree(&dplanet_soa->y[0]);
			cudaFree(&dplanet_soa->vx[0]);
			cudaFree(&dplanet_soa->vy[0]);
			cudaFree(&dplanet_soa->m[0]);
			cudaFree(&dheat[0]);
			cudaFree(&dfx[0]);
			cudaFree(&dfy[0]);
		}


	}
	else {


		float time;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		if (!strcmp(argv[3], "CPU"))
			stepICPU(stepNum);
		else if (!strcmp(argv[3], "OPENMP"))
			stepIOPENMP(stepNum);
		else
			stepICUDA(stepNum);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		printf("Execution time %f milliseconds\n", time);



	}
	free(&ax[0]);
	free(&ay[0]);
	free(planet_soa->x);
	free(planet_soa->y);
	free(planet_soa->vx);
	free(planet_soa->vy);
	free(planet_soa->m);
	free(planet_soa);
	free(dplanet_soa);

	free(&heat[0]);
	return 0;
}



void print_help() {
	printf("nbody_%s N D M [-i I] [-i input_file]\n", USER_NAME);

	printf("where:\n");
	printf("\tN                Is the number of bodies to simulate.\n");
	printf("\tD                Is the integer dimension of the activity grid. The Grid has D*D locations.\n");
	printf("\tM                Is the operation mode, either  'CPU' or 'OPENMP' or 'GPU'\n");
	printf("\t[-i I]           Optionally specifies the number of simulation iterations 'I' to perform. Specifying no value will use visualisation mode. \n");
	printf("\t[-f input_file]  Optionally specifies an input file with an initial N bodies of data. If not specified random data will be created.\n");
}




