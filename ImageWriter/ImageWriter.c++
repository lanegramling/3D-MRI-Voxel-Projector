// ImageWriter.c++ -- Abstract base class for writing image files

#include <iostream>
using namespace std;

#include "ImageWriter.h"

// Known subclasses (needed by factory method "create"):
#include "BMPImageWriter.h"
#include "JPEGImageWriter.h"
#include "PNGImageWriter.h"

ImageWriter::ImageWriter(string fName, int xres, int yres, int numChannels) :
	mImageFileName(fName),
	mXRes(xres), mYRes(yres), mNumChannels(numChannels)
{
}

ImageWriter::ImageWriter(const ImageWriter& rte)
{
	// disallowed!
}

ImageWriter::~ImageWriter()
{
}

ImageWriter* ImageWriter::create(std::string fileName, int xres, int yres, int numChannels)
{
	return guessFileType(fileName, xres, yres, numChannels);
}

ImageWriter* ImageWriter::guessFileType(const std::string& fileName, int xres, int yres, int numChannels)
{
	int dotLoc = fileName.find_last_of('.');
	if (dotLoc != std::string::npos)
	{
		std::string extension = fileName.substr(dotLoc+1);
		if ((extension.compare("bmp") == 0) || (extension.compare("BMP") == 0))
			return new BMPImageWriter(fileName, xres, yres, numChannels);
		if ((extension.compare("jpg") == 0) || (extension.compare("JPG") == 0))
			return new JPEGImageWriter(fileName, xres, yres, numChannels);
		if ((extension.compare("jpeg") == 0) || (extension.compare("JPEG") == 0))
			return new JPEGImageWriter(fileName, xres, yres, numChannels);
		if ((extension.compare("png") == 0) || (extension.compare("PNG") == 0))
			return new PNGImageWriter(fileName, xres, yres, numChannels);
	}
	
	cerr << "ImageWriter::guessFileType cannot determine file type of: "
	     << fileName << endl;
	return nullptr;
}
