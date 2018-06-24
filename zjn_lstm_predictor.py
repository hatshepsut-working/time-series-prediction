import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
from math import sqrt

# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
#alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#alphabet = ["我","今天","没","吃药","感觉","自己","萌萌哒"]
alphabet = []
before = [77, 76, 74, 76, 64, 76, 76, 76, 76, 75, 75, 76, 76, 80, 75, 75, 75, 75, 75, 74, 74, 72, 72, 73, 72, 73, 72, 72, 72, 71, 295, 98, 102, 94, 91, 90, 89, 90, 87, 83, 85, 84, 85, 82, 82, 78, 80, 77, 80, 78, 76, 77, 77, 76, 76, 75, 75, 74, 88, 74, 75, 74, 77, 72, 71, 72, 73, 71, 71, 70, 71, 77, 72, 69, 70, 67, 71, 68, 69, 70, 64, 74, 73, 71, 70, 69, 70, 63, 68, 68, 66, 66, 67, 64, 84, 67, 67, 67, 67, 70, 66, 66, 65, 64, 65, 63, 63, 63, 62, 65, 72, 68, 70, 67, 66, 70, 63, 63, 66, 69, 65, 67, 68, 68, 69, 66, 66, 65, 66, 55, 59, 59, 59, 54, 58, 59, 58, 58, 59, 58, 58, 58, 60, 57, 58, 57, 57, 60, 57, 56, 59, 57, 121, 56, 51, 56, 56, 56, 55, 57, 67, 56, 56, 54, 55, 55, 54, 56, 55, 50, 58, 55, 57, 71, 55, 55, 54, 54, 49, 68, 54, 54, 55, 54, 54, 53, 54, 53, 52, 51, 52, 53, 54, 52, 49, 52, 58, 52, 55, 54, 52, 52, 53, 52, 56, 52, 52, 51, 51, 46, 51, 51, 51, 51, 51, 51, 50, 51, 51, 51, 51, 50, 50, 50, 49, 51, 51, 51, 50, 51, 51, 45, 50, 52, 56, 50, 51, 53, 51, 49, 50, 51, 52, 49, 49, 50, 50, 49, 50, 50, 50, 50, 48, 50, 51, 49, 49, 49, 49, 49, 50, 50, 51, 50, 50, 50, 51, 50, 50, 49, 50, 49, 49, 49, 49, 49, 49, 50, 50, 49, 43, 49, 51, 49, 50, 49, 51, 49, 49, 49, 49, 50, 50, 49, 50, 50, 50, 48, 49, 49, 48, 50, 46, 48, 50, 50, 49, 50, 49, 46, 50, 49, 50, 49, 51, 50, 50, 50, 50, 51, 48, 51, 50, 52, 50, 52, 50, 50, 51, 50, 51, 49, 40, 47, 52, 50, 50, 51, 50, 50, 50, 52, 53, 50, 50, 50, 52, 50, 50, 50]
for i in range(350):
    alphabet.append(before[i])
#alphabet = [27.63, 27.61, 27.6, 27.61, 27.61, 27.64, 27.65, 27.66, 27.67, 27.68, 27.69, 27.7, 27.73, 27.75, 27.76, 27.76, 27.76, 27.79, 27.79, 27.79, 27.79, 27.79, 27.79, 27.78, 27.78, 27.78, 27.78, 27.78, 27.78, 27.78, 27.78, 27.75, 27.72, 27.7, 27.7, 27.68, 27.64, 27.61, 27.58, 27.58, 26.86, 26.87, 26.89, 26.91, 26.89, 26.89, 26.91, 26.92, 26.93, 26.94, 26.95, 26.96, 26.97, 26.98, 26.99, 27.0, 27.01, 27.27, 27.28, 26.67, 26.67, 26.68, 26.65, 26.66, 26.64, 26.67, 26.69, 26.68, 26.65, 26.64, 26.63, 26.58, 26.57, 26.57, 26.55, 26.52, 26.52, 26.49, 26.49, 26.51, 26.5, 26.51, 26.52, 26.51, 26.45, 26.46, 26.43, 26.45, 26.42, 26.42, 26.42, 26.39, 26.4, 26.36, 26.38, 26.4, 26.42, 26.46, 26.42, 26.37, 26.36, 26.32, 26.26, 26.24, 26.23, 26.17, 26.16, 26.15, 26.1, 26.08, 26.08, 26.03, 26.03, 26.03, 26.01, 26.01, 25.97, 25.94, 25.96, 25.95, 25.92, 25.92, 25.9, 25.86, 25.87, 25.84, 25.83, 25.83, 25.83, 25.8, 25.81, 25.8, 25.78, 25.78, 25.78, 25.76, 25.77, 25.76, 25.75, 25.76, 25.74, 25.74, 25.74, 25.72, 25.71, 25.72, 25.69, 25.69, 25.71, 25.65, 25.67, 25.67, 25.65, 25.65, 25.65, 25.63, 25.63, 25.64, 25.61, 25.62, 25.63, 25.6, 25.61, 25.61, 25.59, 25.6, 25.6, 25.58, 25.59, 25.59, 25.58, 25.58, 25.58, 25.56, 25.57, 25.56, 25.54, 25.54, 25.54, 25.53, 25.54, 25.53, 25.53, 25.54, 25.53, 25.51, 25.52, 25.51, 25.5, 25.51, 25.51, 25.5, 25.51, 25.49, 25.48, 25.49, 25.47, 25.47, 25.47, 25.45, 25.46, 25.46, 25.44, 25.45, 25.44, 25.43, 25.44, 25.42, 25.42, 25.43, 25.39, 25.42, 25.39, 25.37, 25.39, 25.37, 25.36, 25.37, 25.35, 25.36, 25.37, 25.35, 25.35, 25.35, 25.33, 25.34, 25.32, 25.31, 25.32, 25.31, 25.31, 25.31, 25.29, 25.3, 25.3, 25.27, 25.28, 25.27, 25.27, 25.28, 25.27, 25.27, 25.25, 25.24, 25.25, 25.24, 25.24, 25.25, 25.22, 25.23, 25.22, 25.23, 25.23, 25.21, 25.22, 25.21, 25.21, 25.2, 25.2, 25.2, 25.18, 25.18, 25.17, 25.17, 25.18, 25.16, 25.17, 25.16, 25.17, 25.16, 25.16, 25.16, 25.15, 25.15, 25.14, 25.15, 25.13, 25.14, 25.14, 25.14, 25.14, 25.12, 25.13, 25.12, 25.13, 25.13, 25.13, 25.12, 25.1, 25.12, 25.09, 25.09, 25.08, 25.09, 25.08, 25.08, 25.08, 25.08, 25.08, 25.08, 25.08, 25.08, 25.08, 25.07, 25.08, 25.07, 25.07, 25.07, 25.06, 25.07, 25.07, 25.08, 25.06, 25.07, 25.06, 25.06, 25.06, 25.06, 25.05, 25.05, 25.06, 25.05, 25.05, 25.05, 25.05, 25.04, 25.05, 25.04, 25.05, 25.05, 25.05, 25.05, 25.05, 25.04, 25.04, 25.04, 25.04, 25.04, 25.04, 25.05, 25.04, 25.04, 25]
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))#i:index c:number
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print (seq_in, '->', seq_out)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), 1, seq_length))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# create and fit the model
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, nb_epoch=1000, batch_size=1, verbose=2)


model.save('./my_model.h5')
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
# demonstrate some model predictions

actual_data = []
for i in range(3, len(alphabet)):
    actual_data.append(alphabet[i])
forecasted_data = []

for pattern in dataX:
    x = numpy.reshape(pattern, (1, 1, len(pattern)))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print (seq_in, "->", result)
    forecasted_data.append(result)

print(len(actual_data))
print(actual_data)
print(len(forecasted_data))
print(forecasted_data)

# evaluation: MSE RMSE MAE
error = []
squaredError = []
absError = []
i = len(actual_data)
for z in range(3, i):
    error.append(actual_data[z] - forecasted_data[z])
for val in error:
    squaredError.append(val * val)
    absError.append(abs(val))

    # MSE：均方误差
print("MSE = ", sum(squaredError) / len(squaredError))

    # RMSE：均方根误差
print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))

    # MAE：平均绝对误差
print("MAE = ", sum(absError) / len(absError))

actual = [i for i in actual_data[1:]]
forecasted = [j for j in forecasted_data[1:]]

plt.plot(range(len(actual)), actual, 'r', range(len(forecasted)), forecasted, 'b')
plt.title("LSTM")
legend = ['actual', 'forecasted']
plt.legend(legend)
plt.show()