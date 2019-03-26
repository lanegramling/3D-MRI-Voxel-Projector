// BMPImageWriter.c++ -- Simple class to build bmp files

#include <iostream>
#include <fstream>
using namespace std;

#include "BMPImageWriter.h"

BMPImageWriter::BMPImageWriter(string fName, int xres, int yres, int nChannels) :
	ImageWriter(fName,xres,yres,nChannels),
	mImageFile(nullptr), mBH(xres,yres,nChannels), mBuffer(nullptr), mDblToUCBuf(nullptr),
	mReportedOpenFailure(false)
{
	mBytesPerLineIN = mBytesPerLineOUT = xres * nChannels; // assumes 24-bit images
	/* round up to a word boundary */
	if (mBytesPerLineOUT & 0x0003) 
	{
		mBytesPerLineOUT |= 0x0003;
		++mBytesPerLineOUT;
	}
	mBuffer = new unsigned char[mBytesPerLineOUT];
	mDblToUCBuf = new unsigned char[mBytesPerLineOUT];
	for (int i=0 ; i<mBytesPerLineOUT ; i++)
		mBuffer[i] = static_cast<unsigned char>(0);
	mBH.filesize = mBH.headersize + mBytesPerLineOUT*mBH.depth;
}

BMPImageWriter::BMPImageWriter(const BMPImageWriter& rte) : ImageWriter(rte),
	mBH(0,0,0), mBuffer(nullptr)
{
	// disallowed!
}

BMPImageWriter::~BMPImageWriter()
{
	closeImageFile();
	if (mBuffer != nullptr)
		delete [] mBuffer;
	if (mDblToUCBuf != nullptr)
		delete [] mDblToUCBuf;
}

void BMPImageWriter::addScanLine(const double* sLine)
{
	if (mDblToUCBuf == nullptr)
		return;

	int next = 0;
	for (int i=0 ; i<mXRes ; i++)
		for (int j=0 ; j<mNumChannels ; j++)
		{
			double v = sLine[next] * 255.0;
			if (v < 0.0)
				mDblToUCBuf[next] = 0;
			else if (v > 255.0)
				mDblToUCBuf[next] = 255;
			else
				mDblToUCBuf[next] = static_cast<unsigned char>(v+0.5);
			next++;
		}
	addScanLine(mDblToUCBuf);
}

void BMPImageWriter::addScanLine(const unsigned char* sLine)
{
	if (mImageFile == nullptr)
		if (!openImageFile())
			return;

	if (mBuffer != nullptr)
		for (int i=0 ; i<mBytesPerLineIN ; i+=mNumChannels)
		{
			// copy and switch from RGB to BGR
			mBuffer[i]   = sLine[i+2];
			mBuffer[i+1] = sLine[i+1];
			mBuffer[i+2] = sLine[i];
			if (mNumChannels == 4)
				mBuffer[i+3] = sLine[i+3];
		}

	mImageFile->write(reinterpret_cast<const char*>(mBuffer), mBytesPerLineOUT);
}

void BMPImageWriter::closeImageFile()
{
	if ((mImageFile != nullptr) && !mImageFile->fail())
		mImageFile->close();
}

bool BMPImageWriter::openImageFile()
{
	mImageFile = new ofstream(mImageFileName.c_str(), ios::out|ios::binary);
	if (mImageFile->fail())
	{
		if (!mReportedOpenFailure)
		{
			cerr << "Could not open " << mImageFileName << " for writing\n";
			mReportedOpenFailure = true;
		}
		return false;
	}

	write8Bits(mBH.id[0]); write8Bits(mBH.id[1]);
	write32Bits(mBH.filesize);
	write16Bits(mBH.reserved[0]); write16Bits(mBH.reserved[1]);
	write32Bits(mBH.headersize);
	write32Bits(mBH.infoSize);
	write32Bits(mBH.width);
	write32Bits(mBH.depth);
	write16Bits(mBH.biPlanes);
	write16Bits(mBH.bits);
	write32Bits(mBH.biCompression);
	write32Bits(mBH.biSizeImage);
	write32Bits(mBH.biXPelsPerMeter);
	write32Bits(mBH.biYPelsPerMeter);
	write32Bits(mBH.biClrUsed);
	write32Bits(mBH.biClrImportant);

	return true;
}

void BMPImageWriter::write8Bits(char val) const
{
	mImageFile->write(&val, 1);
}

void BMPImageWriter::write16Bits(int val) const
{
	write8Bits( (val & 0xff) );
	write8Bits( ((val>>8) & 0xff) );
}

void BMPImageWriter::write32Bits(long val) const
{
	write8Bits( (val & 0xff) );
	write8Bits( ((val>>8) & 0xff) );
	write8Bits( ((val>>16) & 0xff) );
	write8Bits( ((val>>24) & 0xff) );
}

void BMPImageWriter::writeImage(const unsigned char* fb)
{
	const unsigned char* ptr = fb;
	for (int j=0 ; j<mYRes ; j++)
	{
		addScanLine(ptr);
		ptr += mBytesPerLineIN;
	}
}
