// test.c++ -- try out various ImageWriter subclasses

#include <iostream>
using namespace std;
#include <stdlib.h>

#include "ImageWriter.h"

void writeTheFile(std::string fName, int xres, int yres)
{
	unsigned char* row = new unsigned char[3*xres];
	ImageWriter* w = ImageWriter::create(fName,xres,yres);
	for (int r=0 ; r<yres ; r++)
	{
		for (int c=0 ; c<3*xres ; c+=3)
		{
			bool firstHalf = (c < (3*xres/2));
			if (r < 10)
				if (firstHalf)
				{
					row[c] = 255; row[c+1] = 0; row[c+2] = 0;
				}
				else
				{
					row[c] = 0; row[c+1] = 255; row[c+2] = 0;
				}
			else if (r < 20)
				if (firstHalf)
				{
					row[c] = 255; row[c+1] = 255; row[c+2] = 0;
				}
				else
				{
					row[c] = 0; row[c+1] = 255; row[c+2] = 255;
				}
			else
				if (firstHalf)
				{
					row[c] = 0; row[c+1] = 0; row[c+2] = 255;
				}
				else
				{
					row[c] = 255; row[c+1] = 0; row[c+2] = 255;
				}
		}
		w->addScanLine(row);
	}
	w->closeImageFile();
}

void writeTheGrid(std::string fName, int xres, int yres,
	int xWidth, int xSpace, int yWidth, int ySpace)
{
	unsigned char* row = new unsigned char[3*xres];
	ImageWriter* w = ImageWriter::create(fName,xres,yres);
	for (int r=0 ; r<yres ; r++)
	{
		bool pixelRowIsBlack = (r % (yWidth + ySpace) < yWidth);
		for (int c=0 ; c<xres ; c++)
		{
			bool pixelColIsBlack = (c % (xWidth + xSpace) < xWidth);
			if (pixelRowIsBlack || pixelColIsBlack)
				row[3*c] = row[3*c+1] = row[3*c+2] = 0;
			else
				row[3*c] = row[3*c+1] = row[3*c+2] = 255;
		}
		w->addScanLine(row);
	}
	w->closeImageFile();
}

int main(int argc, char* argv[])
{
	if (argc < 4)
		cerr << "Usage: " << argv[0] << "fName xres yres [xW xS yW yS]\n";
	else
	{
		int xres = atoi(argv[2]);
		int yres = atoi(argv[3]);
		if (argc == 4)
		{
			writeTheFile(argv[1], xres, yres);
			writeTheFile(std::string("_")+argv[1], xres+10, yres);
		}
		else if (argc == 8)
		{
			int xW = atoi(argv[4]);
			int xS = atoi(argv[5]);
			int yW = atoi(argv[6]);
			int yS = atoi(argv[7]);
			writeTheGrid(argv[1], xres, yres, xW, xS, yW, yS);
		}
		else
			cerr << "Usage: " << argv[0] << "fName xres yres [xW xS yW yS]\n";
	}
	return 0;
}
