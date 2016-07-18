from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline

data_dim = 14
timesteps = 2
nb_classes = 2
batch_size = 2
rows = 115
signal_num = [3,4,5,6,7,8,9,10,11,12,13,14,15,16]
signal_file = ["rubix.txt"]

for n in signal_file:
	# set up model
	model = Sequential()
	model.add(LSTM(32, return_sequences=True, stateful=False,
	               input_shape=(50, 14)))
	#model.add(Dropout(0.5))
	model.add(LSTM(32, stateful=False))
	#model.add(Dropout(0.5))

	model.add(Dense(2, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	#get focused data
	y_train = np.zeros((rows*2, nb_classes))
	classes = np.zeros(rows*2)
	focusedData = np.loadtxt(signal_file, delimiter=",", skiprows=1)
	t = 0
	x = 0
	focusedSignals = np.zeros((rows, 50, data_dim))
	signal = 0

	for n in signal_num:
	    t = 0
	    x=0
	    x_plus=x+timesteps
	    while t < rows:
	        for c in range(0,50):
	            points = np.mean(focusedData[x:x_plus, n])
	            focusedSignals[t,c,signal] = points
	            x+=2
	            x_plus=x+2
	        t+=1
	    signal+=1
	newFocused = focusedSignals - 4400

	#set one-hot
	for i in range(0,rows):
	    y_train[i][0]=1
	    y_train[i][1]=0
	    classes[i]=1

	#get scared data
	scaredData = np.loadtxt("juggling.txt", delimiter=",", skiprows=1)
	t = 0
	x = 0

	scaredSignals = np.zeros((rows, 50, data_dim))

	signal = 0

	for n in signal_num:
	    t = 0
	    x=0
	    x_plus=x+timesteps
	    while t < rows:
	        for c in range(0,50):
	            points = np.average(scaredData[x:x_plus, n])
	            scaredSignals[t,c,signal] = points
	            x+=2
	            x_plus=x+2
	        t+=1
	    signal+=1
	newScared = scaredSignals - 4400

	#set focues one-hot
	for i in range(rows, rows*2):
	    y_train[i][0]=0
	    y_train[i][1]=1
	    classes[i]=0

	trainFocus = signals[0:95]
	testFocus = signals[95:115]
	trainScared = signals[115:210]
	testScared = signals[210:230]
	totalTrain = np.vstack((trainFocus, trainScared))
	totalTest = np.vstack((testFocus, testScared))

	trainLabel1 = y_train[0:95]
	trainLabel2 = y_train[115:210]
	testLabel1 = y_train[95:115]
	testLabel2 = y_train[210:230]
	train_labels = np.vstack((trainLabel1, trainLabel2))
	test_labels = np.vstack((testLabel1, testLabel2))

	error_and_acc = model.fit(totalTrain, train_labels, shuffle = True, 
	                          batch_size=batch_size, nb_epoch=100, validation_data=(totalTest, test_labels),
	                          verbose = 1)

	var_file = open("data_analysis.txt", w)

	for i in range(len(error_and_acc)):
	    var_file.write(error_and_acc.history["acc"][i])
	var_file.write(signal_file)
	var_file.close()
