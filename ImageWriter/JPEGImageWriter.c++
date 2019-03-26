// JPEGImageWriter.c++ -- Simple class to build jpeg files

#include <iostream>
using namespace std;

#include "JPEGImageWriter.h"

JPEGImageWriter::JPEGImageWriter(string fName, int xres, int yres, int numChannels,
		int quality) :
	ImageWriter(fName,xres,yres,numChannels),
	mImageFile(nullptr), mCompInfo(nullptr), mQuality(quality), mScanLine(nullptr),
	scanLineNoAlphaUC(nullptr),
	mReportedOpenFailure(false)
{
}

JPEGImageWriter::JPEGImageWriter(const JPEGImageWriter& rte) : ImageWriter(rte)
{
	// disallowed!
}

JPEGImageWriter::~JPEGImageWriter()
{
	closeImageFile();
}

void JPEGImageWriter::addScanLine(const double* sLine)
{
	if (mImageFile == nullptr)
		if (!openImageFile())
			return;
	if (mCompInfo == nullptr)
		return;

	int next = 0;
	for (int i=0 ; i<mXRes ; i++)
	{
		for (int j=0 ; j<3 ; j++)
		{
			double v = sLine[next] * 255.0;
			if (v < 0.0)
				mScanLine[next] = 0;
			else if (v > 255.0)
				mScanLine[next] = 255;
			else
				mScanLine[next] = static_cast<unsigned char>(v+0.5);
			next++;
		}
		if (mNumChannels == 4)
			// skip over alpha channel
			next++;
	}
	jpeg_write_scanlines(mCompInfo, &mScanLine, 1);
}

void JPEGImageWriter::addScanLine(const unsigned char* sLine)
{
	if (mImageFile == nullptr)
		if (!openImageFile())
			return;
	if (mCompInfo == nullptr)
		return;

	if (mNumChannels == 4)
	{
		// lose the alpha channel
		if (scanLineNoAlphaUC == nullptr)
			scanLineNoAlphaUC = new unsigned char[3 * mXRes];
		int next = 0;
		for (int i=0 ; i<4*mXRes ; i+=4)
			for (int j=0 ; j<3 ; j++)
				scanLineNoAlphaUC[next++] = sLine[i+j];
		jpeg_write_scanlines(mCompInfo, const_cast<unsigned char**>(&scanLineNoAlphaUC), 1);
	}
	else
		jpeg_write_scanlines(mCompInfo, const_cast<unsigned char**>(&sLine), 1);
}

void JPEGImageWriter::closeImageFile()
{
	if (mCompInfo == nullptr)
		return;

	jpeg_finish_compress(mCompInfo);
	jpeg_destroy_compress(mCompInfo);
	mCompInfo = nullptr;
	if (mScanLine != nullptr)
	{
		delete [] mScanLine;
		mScanLine = nullptr;
	}
	if (mImageFile != nullptr)
	{
		fclose(mImageFile);
		mImageFile = nullptr;
	}
	if (scanLineNoAlphaUC != nullptr)
	{
		delete [] scanLineNoAlphaUC;
	}
}

bool JPEGImageWriter::openImageFile()
{
	mImageFile = fopen(mImageFileName.c_str(), "wb");

	if( !mImageFile )
	{
		if (!mReportedOpenFailure)
		{
			cerr << "Error opening file: " << mImageFileName <<  '\n';
			mReportedOpenFailure = true;
		}
		return false;
	}

	// Allocate and initialize the compression object
	mCompInfo = new struct jpeg_compress_struct();
	struct jpeg_error_mgr jerr;
	mCompInfo->err = jpeg_std_error(&jerr);
	jpeg_create_compress(mCompInfo);

	// tell the object where the output jpeg file is to be written
	jpeg_stdio_dest(mCompInfo, mImageFile);

	// define the image properties (size, colorspace, etc.)
	mCompInfo->image_width = mXRes;
	mCompInfo->image_height = mYRes;
	mCompInfo->input_components = 3; /* # of color components per pixel */
	mCompInfo->in_color_space = JCS_RGB; /* colorspace of input image */
	jpeg_set_defaults(mCompInfo);
	// 27June2006 - documentation says following are defaults, but they don't
	// seem to be (but should be)
	mCompInfo->density_unit = 0;
	mCompInfo->X_density = 1;
	mCompInfo->Y_density = 1;
	jpeg_set_quality(mCompInfo,mQuality,TRUE);

	// JSAMPLE is "unsigned char"
	// 'mScanLine': only used when the "double*" version of addScanLine is used
	mScanLine = new JSAMPLE[mXRes*3];

	jpeg_start_compress(mCompInfo, TRUE);
	return true;
}

void JPEGImageWriter::writeImage(const unsigned char* fb)
{
	int actualRowWidth = mNumChannels * mXRes;
	const unsigned char* ptr = fb + (actualRowWidth * (mYRes - 1));
	for (int j=0 ; j<mYRes ; j++)
	{
		addScanLine(ptr);
		ptr -= actualRowWidth;
	}
}
