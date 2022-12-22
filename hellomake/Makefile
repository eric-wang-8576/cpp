CC = gcc
CFLAGS = -I.
DEPS = hellomake.h
OBJ = hellofunc.o hellomake.o 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

hellomake: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)