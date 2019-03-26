CPP = g++ -std=c++11 -fPIC

CFLAGS = -c

OBJS = ImageWriter.o BMPImageWriter.o JPEGImageWriter.o PNGImageWriter.o

../lib/libCOGLImageWriter.so: libCOGLImageWriter.so
	cp libCOGLImageWriter.so ../lib/

libCOGLImageWriter.so: $(OBJS)
	g++ -shared -o libCOGLImageWriter.so $(OBJS) -lpng -ljpeg

ImageWriter.o: ImageWriter.c++
	$(CPP) $(CFLAGS) ImageWriter.c++

BMPImageWriter.o: BMPImageWriter.c++
	$(CPP) $(CFLAGS) BMPImageWriter.c++

JPEGImageWriter.o: JPEGImageWriter.c++
	$(CPP) $(CFLAGS) JPEGImageWriter.c++

PNGImageWriter.o: PNGImageWriter.c++
	$(CPP) $(CFLAGS) PNGImageWriter.c++
