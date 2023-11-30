# Makefile for gpu benchmark programs
VERSION=		0.5
#CUDA    = /Developer/NVIDIA/CUDA-8.0
CUDA=			$(shell dirname `dirname \`which nvcc\``)

CUDA_INCLUDE=		$(shell dirname `find $(CUDA) -name cuda.h`)
CUDA_LIBDIR=		$(shell dirname `find $(CUDA) -name libcuda.so`|head -n1)
NVRTC_INCLUDE=		$(shell dirname `find $(CUDA) -name nvrtc.h`)
NVRTC_LIBDIR=		$(shell dirname `find $(CUDA) -name libnvrtc.so`|head -n1)
#POWER_SENSOR=		$(HOME)/projects/libpowersensor-master/build
ARCH=			$(shell arch)

# CC must be able to generate dependencies with -MM option
CC      = gcc -O3 -Wall -g
CXX     = g++ #-Wno-deprecated-declarations
NVCC	= nvcc -O3
NVDEBUG = nvcc -lineinfo -g -G

INCLUDES=		-I.
INCLUDES+=		-I$(CUDA_INCLUDE) -I$(NVRTC_INCLUDE)

CXXFLAGS=		-std=c++11 -O3 -g -fpic -fopenmp $(INCLUDES) -DNDEBUG
NVCCFLAGS=		$(INCLUDES)

LIBTCC_SOURCES=		util/cu.cc\
			util/nvrtc.cc\
			libtcc/CorrelatorKernel.cc\
			libtcc/Correlator.cc\
			libtcc/Kernel.cc

CORRELATOR_TEST_SOURCES=test/CorrelatorTest/CorrelatorTest.cc\
			test/CorrelatorTest/Options.cc\
			test/Common/Record.cc\
			test/Common/UnitTest.cc

SIMPLE_EXAMPLE_SOURCES=	test/SimpleExample/SimpleExample.cu

LIBTCC_OBJECTS=		$(LIBTCC_SOURCES:%.cc=%.o) libtcc/TCCorrelator.o
SIMPLE_EXAMPLE_OBJECTS=	$(SIMPLE_EXAMPLE_SOURCES:%.cu=%.o)
CORRELATOR_TEST_OBJECTS=$(CORRELATOR_TEST_SOURCES:%.cc=%.o)

OBJECTS_TCC=		$(LIBTCC_OBJECTS)\
            $(SIMPLE_EXAMPLE_OBJECTS)\
			$(CORRELATOR_TEST_OBJECTS)

SHARED_OBJECTS=		libtcc/libtcc.so libtcc/libtcc.so.$(VERSION)

DEPENDENCIES=		$(LIBTCC_OBJECTS:%.o=%.d)

EXECUTABLES= test/CorrelatorTest/CorrelatorTest

LIBRARIES=		-L$(CUDA_LIBDIR) -lcuda\
			-L$(NVRTC_LIBDIR) -lnvrtc -lnvToolsExt


#EXECS = testgpukernel benchmark_gxkernel validate_xcorr testgpukernel_half

#SOURCES	= gxkernel.cu testgpukernel.cu benchmark_gxkernel.cu validate_xcorr.cu

EXECS = benchmark_gxkernel

SOURCES = gxkernel.cu benchmark_gxkernel.cu

OBJECTS = $(SOURCES:.cu=.o)

CUFLAGS	= -I../../fxkernel/src/ $(INCLUDES)

CFLAGS  = 

CLINK	= -lcudart -lcufft -lcurand -lstdc++ -lm

#%.o : %.cu
#	$(NVCC) $(CUFLAGS) -c $<

#%.o : %.cpp
#	$(CC) $(CFLAGS) -c $<

#%.o : %.c
#	$(CC) -c -O3 -ffast-math $<
#	$(CC) -c  $<


%.d:			%.cc
			-$(CXX) $(CXXFLAGS) -MM -MT $@ -MT ${@:%.d=%.o} -MT ${@:%.d=%.s} $< -o $@

%.d:			%.cu
			-$(CXX) -x c++ $(CXXFLAGS) -MM -MT $@ -MT ${@:%.d=%.o} -MT ${@:%.d=%.s} $< -o $@

%.o:			%.cc
			$(CXX) $(CXXFLAGS) -o $@ -c $<

%.o:			%.cu
			$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.s:			%.cc
			$(CXX) $(CXXFLAGS) -o $@ -S $<

%.so:			%.so.$(VERSION)
			rm -f $@
			ln -s $(@F).$(VERSION) $@


all: $(EXECS) $(EXECUTABLES)

libtcc/TCCorrelator.o:	libtcc/TCCorrelator.cu	# CUDA code embedded in object file
			ld -r -b binary -o $@ $<

libtcc/TCCorrelator.d:
			-

libtcc/libtcc.so.$(VERSION):		$(LIBTCC_OBJECTS)
			$(CXX) -shared -o $@ -Wl,-soname=$@ $^ $(LIBRARIES)

test/SimpleExample/SimpleExample:		$(SIMPLE_EXAMPLE_OBJECTS) libtcc/libtcc.so
			$(NVCC) $(NVCCFLAGS) -o $@ $(SIMPLE_EXAMPLE_OBJECTS) -Xlinker -rpath=. -Llibtcc -ltcc $(LIBRARIES)

test/CorrelatorTest/CorrelatorTest:	$(CORRELATOR_TEST_OBJECTS) libtcc/libtcc.so
			$(CXX) $(CXXFLAGS) -o $@ $(CORRELATOR_TEST_OBJECTS) -Wl,-rpath=. -Llibtcc -ltcc $(LIBRARIES)

benchmark_gxkernel: gxkernel.cu benchmark_gxkernel.cu libtcc/libtcc.so
	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^ -Llibtcc -ltcc $(LIBRARIES)

debug: NVCC = $(NVDEBUG)
debug: all

#testgpukernel : gxkernel.cu testgpukernel.cu ../../fxkernel/src/common.cpp
#	$(NVCC) -DUSING_CUDA $(CUFLAGS) $(CLINK) -o $@ $^

#testgpukernel_half : gxkernel.cu testgpukernel.cu ../../fxkernel/src/common.cpp
#	$(NVCC) -DUSING_CUDA  -DUSEHALF $(CUFLAGS) -arch=sm_75 $(CLINK) -o $@ $^

#benchmark_gxkernel : gxkernel.cu benchmark_gxkernel.cu libtcc/libtcc.so
#	$(NVCC) $(CUFLAGS) $(CLINK) -o $@ $^ -rpath=. -Llibtcc -ltcc $(LIBRARIES)




#validate_xcorr : gxkernel.cu validate_xcorr.cu ../../fxkernel/src/common.cpp
#	$(NVCC) -DUSING_CUDA $(CUFLAGS) $(CLINK) -o $@ $^

clean:
	rm -f $(DEP)
	rm -f *.linkinfo
	rm -f $(OBJECTS)
	rm -f $(EXECS)
	rm -f $(OBJECTS_TCC)
	rm -f $(SHARED_OBJECTS) 
	rm -f $(DEPENDENCIES) 
	rm -f $(EXECUTABLES)

