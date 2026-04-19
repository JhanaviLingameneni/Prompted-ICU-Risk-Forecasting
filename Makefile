CC = gcc

# Support both Windows and Mac/Linux environments
ifeq ($(OS),Windows_NT)
    TARGET = score.exe
    RM = del /f
    RUN = $(TARGET)
else
    TARGET = score
    RM = rm -f
    RUN = ./$(TARGET)
endif

all: $(TARGET)

$(TARGET): code/score.c
	$(CC) $(CFLAGS) -o $(TARGET) code/score.c -lm

clean:
	$(RM) $(TARGET)

score_ann:
	$(RUN) code/models/ANN_results.csv data/outcomes/outcomes-b.txt

score_grad_boost:
	$(RUN) code/models/GradientBoosting_results.csv data/outcomes/outcomes-b.txt

score_lstm:
	$(RUN) code/models/LSTM_results.csv data/outcomes/outcomes-b.txt