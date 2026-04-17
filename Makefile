CC = gcc
TARGET = score.o
all: $(TARGET)

$(TARGET): code/score.c
	$(CC) $(CFLAGS) -o $(TARGET) code/score.c -lm

clean:
	rm -f $(TARGET)

score_ann:
	./$(TARGET) code/models/ANN_results.csv data/outcomes/outcomes-b.txt

score_grad_boost:
	./$(TARGET) code/models/GradientBoosting_results.csv data/outcomes/outcomes-b.txt

score_lstm:
	./$(TARGET) code/models/LSTM_results.csv data/outcomes/outcomes-b.txt