// PNGImageWriter.h -- Simple class to build PNG files.

#ifndef PNGIMAGEWRITER_H
#define PNGIMAGEWRITER_H

#include <stdio.h>
#include <png.h>

#include "Packed3DArray.h"

#include "ImageWriter.h"

class PNGImageWriter : public ImageWriter
{
public:
	PNGImageWriter(std::string fName, int xres, int yres, int nChannels=3);
	~PNGImageWriter();

	void	closeImageFile();
	
	// Image files can be written EITHER scan line by scan line
	// using addScanLine....

	// all versions of "addScanLine" asssume an array of length:
	//     this->ImageWriter::mNumChannels*this->ImageWriter::mXRes
	// for mNumChannels==3:   RGBRGBRGB...RGB
	void	addScanLine(const double* sLine);
	void	addScanLine(const unsigned char* sLine);

	// ....OR all at once using:
	void	writeImage(const unsigned char* fb);

protected:

private:
	PNGImageWriter(const PNGImageWriter& rte); // disallow copy constructor
	
	// png file pointer and data structures
	FILE* fp;
	png_structp png_ptr;
	png_infop info_ptr;
	cryph::Packed3DArray<unsigned char>* theImage;
	png_bytep* row_pointers;
	int nextScanLine; // only used for line-by-line writing

	bool	openImageFile();
};

#endif
