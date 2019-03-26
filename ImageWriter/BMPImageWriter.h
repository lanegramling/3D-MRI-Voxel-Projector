// BMPImageWriter.h -- Simple class to build BMP files. Based on material found at:
//
// http://www.siggraph.org/education/materials/HyperVis/asp_data/compimag/bmpfile.htm
//
// Notable item not discussed at this site: You really need to write non-character data
// one byte at a time, hence "write8Bits" et al. methods below.

#ifndef BMPIMAGEWRITER_H
#define BMPIMAGEWRITER_H

#include <fstream>

#include "ImageWriter.h"

class BMPImageWriter : public ImageWriter
{
public:
	BMPImageWriter(std::string fName, int xres, int yres, int nChannels=3);
	~BMPImageWriter();

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
	BMPImageWriter(const BMPImageWriter& rte); // disallow copy constructor

	struct BMPHEAD
	{
		char id[2];
		long filesize;
		int reserved[2];
		long headersize;
		long infoSize;
		long width;
		long depth;
		short biPlanes;
		short bits;
		long biCompression;
		long biSizeImage;
		long biXPelsPerMeter;
		long biYPelsPerMeter;
		long biClrUsed;
		long biClrImportant;

		BMPHEAD(int w, int h, int nChannels): filesize(0L),
			// following two work for 24 bit images
			headersize(54), infoSize(40),
			width(w), depth(h), 
			// following two work for 24 bit images
			biPlanes(1), bits(8*nChannels),
			biCompression(0), biSizeImage(0),
			biXPelsPerMeter(0), biYPelsPerMeter(0),
			biClrUsed(0), biClrImportant(0)
		{
			id[0] = 'B'; id[1] = 'M';
			reserved[0] = reserved[1] = 0;
		}
	};

	bool	openImageFile();

	void	write8Bits(char val) const;
	void	write16Bits(int val) const;
	void	write32Bits(long val) const;

	std::ofstream*	mImageFile;
	BMPHEAD			mBH;
	long			mBytesPerLineIN, mBytesPerLineOUT;
	unsigned char*	mBuffer; // used since (i) bytes stored as BGR, and (ii) may need
	                         // to round up to next double word boundary.
	unsigned char*	mDblToUCBuf; // used by addScanLine(const double*)
	bool			mReportedOpenFailure;
};

#endif
