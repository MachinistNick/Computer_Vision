# comparison of training-set based pixel scaling methods on MNIST
from numpy import mean
from numpy import std
from matplotlib import pyplot
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten


'''
#GPU disable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

'''


'''

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
'''




#because of below code GPU start working
import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#Above 5 lines of code for GPU 







# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	width, height, channels = trainX.shape[1], trainX.shape[2], 1
	trainX = trainX.reshape((trainX.shape[0], width, height, channels))
	testX = testX.reshape((testX.shape[0], width, height, channels))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# normalize images
def prep_normalize(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# center images
def prep_center(train, test):
	# convert from integers to floats
	train_cent = train.astype('float32')
	test_cent = test.astype('float32')
	# calculate statistics
	m = train_cent.mean()
	# center datasets
	train_cent = train_cent - m
	test_cent = test_cent - m
	# return normalized images
	return train_cent, test_cent

# standardize images
def prep_standardize(train, test):
	# convert from integers to floats
	train_stan = train.astype('float32')
	test_stan = test.astype('float32')
	# calculate statistics
	m = train_stan.mean()
	s = train_stan.std()
	# center datasets
	train_stan = (train_stan - m) / s
	test_stan = (test_stan - m) / s
	# return normalized images
	return train_stan, test_stan

# repeated evaluation of model with data prep scheme
def repeated_evaluation(datapre_func, n_repeats=10):
	# prepare data
	trainX, trainY, testX, testY = load_dataset()
	# repeated evaluation
	scores = list()
	for i in range(n_repeats):
		# define model
		model = define_model()
		# prepare data
		prep_trainX, prep_testX = datapre_func(trainX, testX)
		# fit model
		model.fit(prep_trainX, trainY, epochs=5, batch_size=64, verbose=0)
		# evaluate model
		_, acc = model.evaluate(prep_testX, testY, verbose=0)
		# store result
		scores.append(acc)
		print('> %d: %.3f' % (i, acc * 100.0))
	return scores

all_scores = list()
# normalization
scores = repeated_evaluation(prep_normalize)
print('Normalization: %.3f (%.3f)' % (mean(scores), std(scores)))
all_scores.append(scores)
# center
scores = repeated_evaluation(prep_center)
print('Centered: %.3f (%.3f)' % (mean(scores), std(scores)))
all_scores.append(scores)
# standardize
scores = repeated_evaluation(prep_standardize)
print('Standardized: %.3f (%.3f)' % (mean(scores), std(scores)))
all_scores.append(scores)
# box and whisker plots of results
pyplot.boxplot(all_scores, labels=['norm', 'cent', 'stan'])
pyplot.show()
