CXXFLAGS = -I.
DEPS = chatbot.hpp
OBJ = chatbot.o

USETHREE := 0

ifeq ($(USETHREE), 1) 
	CXXFLAGS += -DUSETHREE
endif

%.o: %.c $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

chatbot: $(OBJ)
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm *.o chatbot