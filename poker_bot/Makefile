CXXFLAGS =-std=c++17
DEPS = bot.hpp
OBJ = bot.o

USETHREE := 0


%.o: %.c $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

bot: $(OBJ)
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm *.o bot
