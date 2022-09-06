#----------------------------------------

HIPCC    = hipcc
HIPFLAGS = -fopenmp --amdgpu-target=gfx908

#----------------------------------------

average_array_elements: average_array_elements.o
	${HIPCC} ${HIPFLAGS} average_array_elements.o -o average_array_elements

average_array_elements.o: average_array_elements.cpp
	${HIPCC} ${HIPFLAGS} -c average_array_elements.cpp

# Clean-up targets --------------------------------------------------------
.PHONY: clean

clean:
	rm -f average_array_elements *.o
