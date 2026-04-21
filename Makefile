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

$(TARGET): code/scripts/score.c
	$(CC) $(CFLAGS) -o $(TARGET) code/scripts/score.c -lm

clean:
	$(RM) $(TARGET)

score_ann:
	@echo "Scoring ANN predictions..." && \
		$(RUN) code/artifacts/results/ANN_predictions.csv data/outcomes/outcomes-b.txt

score_grad_boost:
	@echo "Scoring Gradient Boosting predictions..." && \
		$(RUN) code/artifacts/results/GradientBoosting_predictions.csv data/outcomes/outcomes-b.txt

score_lstm:
	@echo "Scoring LSTM predictions..." && \
		$(RUN) code/artifacts/results/LSTM_predictions.csv data/outcomes/outcomes-b.txt

score_all: score_ann score_grad_boost score_lstm