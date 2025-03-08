CC = gcc
CFLAGS = -framework OpenCL
TARGET = exercise1_matrix_addition

all: $(TARGET)

$(TARGET): exercise1_matrix_addition.c
	$(CC) -o $(TARGET) exercise1_matrix_addition.c $(CFLAGS)

clean:
	rm -f $(TARGET)
