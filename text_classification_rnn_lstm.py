import tensorflow as tf
from tensorflow.keras import layers, models, datasets, preprocessing
# Hyperparameters
max_features = 20000
maxlen = 200
batch_size = 32
embedding_dims = 16
hidden_units = 64
dropout_rate = 0.5
epochs = 20
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
# RNN Model
model = models.Sequential()
model.add(layers.Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(layers.SimpleRNN(hidden_units, activation='tanh'))
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('RNN Train accuracy:', train_acc)
print('RNN Test accuracy:', test_acc)
# LSTM model
model_lstm = models.Sequential()
model_lstm.add(layers.Embedding(max_features, embedding_dims, input_length=maxlen))
model_lstm.add(layers.LSTM(hidden_units, activation='tanh'))
model_lstm.add(layers.Dropout(dropout_rate))
model_lstm.add(layers.Dense(1, activation='sigmoid'))
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.summary()
model_lstm.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
train_loss_lstm, train_acc_lstm = model_lstm.evaluate(x_train, y_train)
test_loss_lstm, test_acc_lstm = model_lstm.evaluate(x_test, y_test)
print('LSTM Train accuracy:', train_acc_lstm)
print('LSTM Test accuracy:', test_acc_lstm)