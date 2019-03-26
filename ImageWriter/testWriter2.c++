// test.c++ -- try out JPEGImageWriter

#include <iostream>
#include <fstream>
#include <cstdlib>
using namespace std;

#include "JPEGImageWriter.h"

int main(int argc, char* argv[])
{
	if (argc < 3)
		cerr << "Usage: " << argv[0] << "inputfile outputfile\n";
	else
	{
		ifstream is(argv[1]);
		if (!is.good())
		{
			std::cerr << "Could not open " << argv[1] << '\n';
			exit(1);
		}
		bool makingGray = false;
		int xres, yres;
		is >> xres >> yres;
		unsigned char* row = new unsigned char[3*xres];
		ImageWriter* w = ImageWriter::create(argv[2],xres,yres);
		for (int i=0 ; i<yres ; i++)
		{
			int in = 0;
			for (int j=0 ; j<xres ; j++)
			{
				unsigned int x;
				if (makingGray)
					is >> x;
				for (int k=0 ; k<3 ; k++)
				{
					if (!makingGray)
						is >> x;
					row[in++] = x;
				}
			}
			w->addScanLine(row);
		}
		w->closeImageFile();
	}
	return 0;
}
