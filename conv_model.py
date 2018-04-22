from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, Embedding, LSTM, Activation
from keras.models import Sequential
from keras.regularizers import l2
from keras import optimizers

def build_conv(kernel_len1, kernel_len2):
    model = Sequential()
    #print model.output_shape
    model.add(Conv1D(30, kernel_len1, input_shape=(147, 4), activation='relu'))
    #model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    #print model.output_shape
    model.add(Conv1D(60, kernel_len2, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Conv1D(90, kernel_len2, activation='relu'))
    # model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Conv1D(120, kernel_len2, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Conv1D(120, 3, activation='relu'))
    # model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Conv1D(120, 3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(emb_size):
    beta = 1e-3
    model = Sequential()
    model.add(Embedding(4, emb_size, input_length=147))
    model.add(LSTM(50, return_sequences=True, W_regularizer=l2(beta), dropout_U=0.1))#, dropout_U=0.5, dropout_W=0.5))
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(LSTM(50, return_sequences=True, W_regularizer=l2(beta), dropout_U=0.1))
    #model.add(LSTM(50, return_sequences=True, dropout_U=0.5, dropout_W=0.5))
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    #model.add(Dense(128, activation='relu'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model

def build_conv_lstm(filters, kernel_len, lstm_hidden_size):
    beta = 1e-3
    model = Sequential()
    
    model.add(Conv1D(filters=50, kernel_size=3, input_shape=(151, 4), kernel_regularizer=l2(beta), padding='same'))
    #model.add(Conv1D(filters, kernel_len, input_shape=(151, 4), kernel_regularizer=l2(beta), padding='same'))
    model.add(Activation('relu'))    
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    
    model.add(LSTM(units=50, return_sequences=True, kernel_regularizer=l2(beta), recurrent_dropout=0.1))
    model.add(Dropout(0.5))
    
    model.add(Flatten())    
    #model.add(Dense(1, kernel_regularizer=l2(beta), activation='sigmoid'))
    model.add(Dense(150, kernel_regularizer=l2(beta), activation='relu'))
	
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_regularizer=l2(beta), activation='sigmoid'))

    optim = optimizers.Adam(lr=0.0003)
    model.compile(optimizer=optim, loss='binary_crossentropy')

    return model
