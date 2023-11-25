CXXFLAGS =-std=c++17
DEPS = main.hpp
OBJ = main.o

%.o: %.c $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

main: $(OBJ)
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm *.o main
