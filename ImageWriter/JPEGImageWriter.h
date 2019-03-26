// JPEGImageWriter.h -- Simple class to build jpeg files

#ifndef JPEGIMAGEWRITER_H
#define JPEGIMAGEWRITER_H

#include <stdio.h>

#include "ImageWriter.h"

#include "jpeglib.h"

class JPEGImageWriter : public ImageWriter
{
public:
	JPEGImageWriter(std::string fName, int xres, int yres, int nChannels=3,
		int quality=90);
	~JPEGImageWriter();

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
	JPEGImageWriter(const JPEGImageWriter& rte); // disallow copy constructor

	bool	openImageFile();

	FILE*							mImageFile;
	struct jpeg_compress_struct*	mCompInfo;
	int								mQuality;
	unsigned char*					mScanLine; // only used as a buffer for
						                // "double*" version of addScanLine

	// following only used if trying to write alpha channel - need to dump alpha
	unsigned char* scanLineNoAlphaUC;
	bool	mReportedOpenFailure;
};

#endif
