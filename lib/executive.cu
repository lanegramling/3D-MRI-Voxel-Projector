#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <math.h> //For round()


#include "ImageWriter.h"


// MAX kernel - Computes results of MAX image into device buffer, in
// as well as the working buffer for the SUM computation during traversal.
__global__
void maxImage(unsigned char* d_voxelData, unsigned char* d_maxBuf, float* d_workBuf, float* d_maxWeightedSum, int zDepth)
{
	//Each thread receives a vector that will be reduced to a single pixel and
	// placed in the buffer. Operate on the given thread's work assignment within
	// the voxel buffer using the ID and offset calculation.


	//Calculate the xy offset of the vector to traverse
	int xy_offset = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned char max = d_voxelData[xy_offset];
	float sum = d_voxelData[xy_offset];

	//Perform computations.
	for (int z_offset=1; z_offset < zDepth; z_offset++) {
		int i = z_offset * blockDim.x * gridDim.x + xy_offset;
		sum += (d_voxelData[i] + 1) / zDepth; //Sum, weighted for distance to front.
		if (max < d_voxelData[i]) max = d_voxelData[i];
	}
	atomicMax((int*)d_maxWeightedSum, (int)sum); //Atomically update the max weighted sum if necessary.

	d_maxBuf[xy_offset] = max;
	d_workBuf[xy_offset] = sum;
}

// SUM kernel - Computes results of SUM image into device buffer by
// normalizing the working buffer computed in the MAX Kernel.
__global__
void sumImage(unsigned char* d_voxelData, unsigned char* d_sumBuf, float* d_workBuf, float* d_maxWeightedSum, int zDepth)
{
	int xy_offset = blockIdx.x * blockDim.x + threadIdx.x;

	d_sumBuf[xy_offset] = ((d_workBuf[xy_offset] / *d_maxWeightedSum) * 255.0);
}


// int project(int projectionType, int xSize, int ySize, int zSize)
// {
// 	//Traversal directions are determined based on projection type.
//
//
// }


void writeImage(std::string fName, int xres, int yres, const unsigned char* imageBytes)
{
	unsigned char* row = new unsigned char[3*xres];
	ImageWriter* w = ImageWriter::create(fName,xres,yres);
	int next = 0;
	for (int r=0 ; r<yres ; r++)
	{
		for (int c=0 ; c<3*xres ; c+=3)
		{
			row[c] = row[c+1] = row[c+2] = imageBytes[next++];
		}
		w->addScanLine(row);
	}
	w->closeImageFile();
	delete w;
	delete [] row;
}


//Used for projection. Orients comparison for whether descending or ascending.
bool cmp(int num, int boundary)
{ return (!boundary) ? num > 0 : num < boundary; }

//main
int main(int argc, char* argv[])
{
	//Report versions
	int driverVersion, runtimeVersion;
	cudaError_t dv = cudaDriverGetVersion(&driverVersion);
	cudaError_t rv = cudaRuntimeGetVersion(&runtimeVersion);
	std::cout << "Driver version: " << driverVersion << "; Runtime version: " << runtimeVersion << "\n\n";


	//Grab and validate input
	if (argc != 7) {
		std::cout << "\nERROR - Incorrect syntax.\nUsage:\n   ./executive nRows nCols nSheets fileName projectionType outputFileNameBase\n";
		exit(1);
	}
	int nRows = std::atoi(argv[1]);
	int nCols = std::atoi(argv[2]);
	int nSheets = std::atoi(argv[3]);
	char* fileName = argv[4];
	int projectionType = std::atoi(argv[5]);
	char* outputFileNameBase = argv[6];


	//Orient according to projection.
	// array indices are always ordered as x, y, z
	int dir[3];
	int start[3];
	int end[3];
	switch(projectionType) {
		case 1:	//Traverse SHEET (min -> max)
			dir[0] = 1;
			dir[1] = 1;
			dir[2] = 1;
			start[0] = 0;
			start[1] = 0;
			start[2] = 0;
			end[0] = nCols;
			end[1] = nRows;
			end[2] = nSheets;
			break;
		case 2: //Traverse SHEET (max -> min)
			dir[0] = -1;
			dir[1] = 1;
			dir[2] = -1;
			start[0] = nCols;
			start[1] = 0;
			start[2] = nSheets;
			end[0] = 0;
			end[1] = nRows;
			end[2] = 0;
			break;
		case 3: //Traverse   COL (max -> min)
			dir[0] = 1;
			dir[1] = 1;
			dir[2] = -1;
			start[0] = 0;
			start[1] = 0;
			start[2] = nCols;
			end[0] = nSheets;
			end[1] = nRows;
			end[2] = 0;
			break;
		case 4: //Traverse   COL (min -> max)
			dir[0] = -1;
			dir[1] = 1;
			dir[2] = 1;
			start[0] = nSheets;
			start[1] = 0;
			start[2] = 0;
			end[0] = 0;
			end[1] = nRows;
			end[2] = nCols;
			break;
		case 5: //Traverse   ROW (max -> min)
			dir[0] = 1;
			dir[1] = -1;
			dir[2] = -1;
			start[0] = 0;
			start[1] = nSheets;
			start[2] = nRows;
			end[0] = nCols;
			end[1] = 0;
			end[2] = 0;
			break;
		case 6: //Traverse   ROW (min -> max)
			dir[0] = 1;
			dir[1] = -1;
			dir[2] = 1;
			start[0] = 0;
			start[1] = nSheets;
			start[2] = 0;
			end[0] = nCols;
			end[1] = 0;
			end[2] = nRows;
			break;
		default: break;
	}

	//(x,y,z Sizes are reflected the non-zero boundary in the respective index)
	int xSize = (start[0]) ? start[0] : end[0];
	int ySize = (start[1]) ? start[1] : end[1];
	int zSize = (start[2]) ? start[2] : end[2];

	//Initialize array
	unsigned char* h_voxelData;
	int size = nRows * nCols * nSheets;
	h_voxelData = new unsigned char[size];

	//Read in h_voxelData from raw file
	printf("Reading %s file...\n", fileName);
	std::ifstream rawFile(fileName);
	rawFile.read(reinterpret_cast<char*>(h_voxelData), size);
  rawFile.close();

	//Project voxel grid.
	int i = 0;
	unsigned char* h_voxel_oriented = new unsigned char[size];
	for (int x=start[0]; cmp(x, end[0]); x+=dir[0])
		for (int y=start[1]; cmp(y, end[1]); y+=dir[1])
			for (int z=start[2]; cmp(z, end[2]); z+=dir[2]) {
				h_voxel_oriented[z*xSize*ySize + y*xSize + x] = h_voxelData[i];
				i++;
			}



	//Copy voxel data to GPU.
	printf("Copying voxel data to GPU buffer...\n");
	unsigned char* d_voxelData;
	size_t voxel_bufSize = size * sizeof(unsigned char);
	cudaMalloc((void**)&d_voxelData, voxel_bufSize);
	cudaMemcpy(d_voxelData, h_voxel_oriented, voxel_bufSize, cudaMemcpyHostToDevice);


	//Allocate image buffers on host.
	printf("Allocating Host buffers...\n");
	int projectionSize = xSize * ySize;
	unsigned char* h_maxBuf = new unsigned char[projectionSize];
	unsigned char* h_sumBuf = new unsigned char[projectionSize];
	float* h_workBuf = new float[projectionSize];

	//Allocate a location on the GPU to store the maxWeightedSum
	float* d_maxWeightedSum;
	cudaMalloc((void**)&d_maxWeightedSum, sizeof(float));

	//Allocate GPU buffers for images in device memory.
	printf("Allocating image buffers on GPU...\n");
	size_t imageBufSize = projectionSize * sizeof(unsigned char);
	unsigned char* d_maxBuf;	//Stores result for MAX image.
	cudaMalloc((void**)&d_maxBuf, imageBufSize);
	unsigned char* d_sumBuf; //Stores result for SUM image.
	cudaMalloc((void**)&d_sumBuf, imageBufSize);
	float* d_workBuf; //Working buffer for SUM computation.
	cudaMalloc((void**)&d_workBuf, projectionSize * sizeof(int));

	//Compute threads/block and blocks/grid.
	// NOTE: I did explore using the occupancy calculation to maximize throughput/
	// utilization, but struggled visualizing how to work with it and returned to this approach
	int blocksPerGrid = xSize;
	int threadsPerBlock = ySize;

	printf("\nPreparing to invoke kernels - Reviewing calculated parameters:\n");
	printf("Size: %d\n", size);
	printf("Blocks per Grid: %d\n", blocksPerGrid);
	printf("Threads per Block: %d\n", threadsPerBlock);
	printf("voxel_bufSize: %d\n", (int)voxel_bufSize);
	printf("\n");

	//Invoke MAX image Kernel.
	maxImage<<<blocksPerGrid, threadsPerBlock>>>(d_voxelData, d_maxBuf, d_workBuf, d_maxWeightedSum, zSize);
	cudaThreadSynchronize();

	//Invoke SUM image Kernel.
	sumImage<<<blocksPerGrid, threadsPerBlock>>>(d_voxelData, d_sumBuf, d_workBuf, d_maxWeightedSum, zSize);
	cudaThreadSynchronize();

	// Copy resulting MAX and SUM images from device memory to host memory
	printf("Copying results back to host...\n");
	cudaMemcpy(h_maxBuf, d_maxBuf, imageBufSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sumBuf, d_sumBuf, imageBufSize, cudaMemcpyDeviceToHost);

	// Free device memory
	printf("Freeing device memory...\n");
	cudaFree(d_voxelData);
	cudaFree(d_maxBuf);
	cudaFree(d_sumBuf);
	cudaFree(d_workBuf);
	cudaFree(d_maxWeightedSum);

	//Write the output images.
	printf("Writing images...\n");
	char fName[(sizeof(outputFileNameBase) + 7*sizeof(char))/sizeof(char)];
	sprintf(fName, "%s%s", outputFileNameBase, "MAX.png");
	writeImage(fName, xSize, ySize, h_maxBuf);
	sprintf(fName, "%s%s", outputFileNameBase, "SUM.png");
	writeImage(fName, xSize, ySize, h_sumBuf);


	//Free up allocated memory from host.
	printf("Freeing memory from host...\n");
	delete[] h_voxelData;
	delete[] h_voxel_oriented;
	delete[] h_maxBuf;
	delete[] h_sumBuf;
	delete[] h_workBuf;
	return 0;
}
