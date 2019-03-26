// PNGImageWriter.c++ -- Simple class to build png files

#include <iostream>
#include <fstream>
using namespace std;

#include "PNGImageWriter.h"

PNGImageWriter::PNGImageWriter(string fName, int width, int height, int nChannels) :
	ImageWriter(fName,width,height,nChannels), fp(nullptr), png_ptr(nullptr), info_ptr(nullptr),
	theImage(nullptr), row_pointers(nullptr), nextScanLine(0)
{
	fp = fopen(fName.c_str(), "wb");
	if (fp == nullptr)
	{
		cerr << "Could not open " << fName << " for writing.\n";
		return;
	}
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	if (png_ptr == nullptr)
	{
		cerr << "Could not create png_ptr\n";
		return;
	}
	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == nullptr)
	{
		png_destroy_write_struct(&png_ptr, nullptr);
		png_ptr = nullptr;
		cerr << "Could not create png_info\n";
		return;
	}
	png_init_io(png_ptr, fp);
	// specify the header information
	const int bitDepth = 8; // for now... (see also uses of "255" and "unsigned char")
	int color_type = PNG_COLOR_TYPE_RGB;
	if (nChannels == 4)
		color_type = PNG_COLOR_TYPE_RGB_ALPHA;
	png_set_IHDR(png_ptr, info_ptr, width, height, bitDepth, color_type,
				 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
				 PNG_FILTER_TYPE_DEFAULT);
	// need to flip order of the rows:
	theImage = new cryph::Packed3DArray<unsigned char>(height, width, nChannels);
	unsigned char* p = theImage->getModifiableData();
	row_pointers = new png_byte*[height];
	int rpi = height;
	for (int i=0 ; i<height ; i++)
		row_pointers[--rpi] = p + i*width*nChannels*sizeof(unsigned char);
	png_set_rows(png_ptr, info_ptr, row_pointers);
}

PNGImageWriter::PNGImageWriter(const PNGImageWriter& rte) :
	ImageWriter("",0,0), fp(nullptr), png_ptr(nullptr), info_ptr(nullptr),
	theImage(nullptr), row_pointers(nullptr)
{
	// disallowed!
}

PNGImageWriter::~PNGImageWriter()
{
	closeImageFile();
	if (row_pointers != nullptr)
		delete [] row_pointers;
	if (theImage != nullptr)
		delete theImage;
}

void PNGImageWriter::addScanLine(const double* sLine)
{
	if (theImage == nullptr)
		return;
	int pos = 0;
	for (int col=0 ; col<mXRes ; col++)
		for (int channel=0 ; channel<mNumChannels ; channel++)
		{
			unsigned char x = static_cast<unsigned char>(sLine[pos++]*255.0 + 0.5);
			theImage->setDataElement(nextScanLine, col, channel, x);
		}
	nextScanLine++;
}

void PNGImageWriter::addScanLine(const unsigned char* sLine)
{
	if (theImage == nullptr)
		return;
	int pos = 0;
	for (int col=0 ; col<mXRes ; col++)
		for (int channel=0 ; channel<mNumChannels ; channel++)
			theImage->setDataElement(nextScanLine, col, channel, sLine[pos++]);
	nextScanLine++;
}

void PNGImageWriter::closeImageFile()
{
	if (fp != nullptr)
	{
		// first write the buffered-up file, then close it.
		const int TRANSFORM_BIT_MASK = 0;
		png_write_png(png_ptr, info_ptr, TRANSFORM_BIT_MASK, nullptr);
		png_write_end(png_ptr, info_ptr);
		fclose(fp);
		fp = nullptr;
		png_destroy_write_struct(&png_ptr, &info_ptr);
	}
}

bool PNGImageWriter::openImageFile()
{
	return true;
}

void PNGImageWriter::writeImage(const unsigned char* fb)
{
	if (theImage == nullptr)
		return;
	int pos = 0;
	for (int row=0 ; row<mYRes ; row++)
		for (int col=0 ; col<mXRes ; col++)
			for (int channel=0 ; channel<mNumChannels ; channel++)
				theImage->setDataElement(row, col, channel, fb[pos++]);
}
