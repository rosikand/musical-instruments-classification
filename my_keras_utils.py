"""
File: my_keras_utils.py 
-------------------
In this file, I store several useful utility functions
I created for deep learning experiments using Keras.  
"""

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg


def my_predict(model, image_path, img_size, classes):
	"""
	Takes in a file path for a singular image and a corresponding
	class list and returns a prediction for that image. 
	Parameters: 
		- model (Keras model): the Keras model that is to be used for the prediction 
		- image_path (str): the path of the image
		- classes (list): a list of classes used in the Keras classification model 
	Returns: 
		- Predicted class 
		- Probability of prediction 
	"""
	show_image(image_path)
	preds = model.predict([prepare_color(image_path, img_size)])
	max_prob = np.max(preds)
	max_prob_index = np.argmax(preds)
	final_probability = max_prob * 100
	percentage = round(final_probability, 2)
	final_prediction = classes[max_prob_index]
	bolded_pred = "\033[1m" + str(final_prediction) + "\033[0m"
	bolded_prob = "\033[1m" + str(final_probability) + "\033[0m"
	p_string = ("The class is " + str(bolded_pred) + " predicted with a probability of " + str(bolded_prob) + " which rounds to " + str(percentage) + "%")  
	print(p_string)
	return final_prediction, final_probability



def prepare_color(filepath, img_size):
	"""
	Helper function for my_predict for RGB images that have
	same width and height for its dimensions. 
	Parameters: 
		- image_path (str): the path of the image
		- img_size: the target size of the image 
	Returns: 
		- Transformed image that is of suitable type to be inserted into a Keras model 
	"""
	img_array = cv2.imread(filepath)
	new_array = cv2.resize(img_array, (img_size, img_size))
	return new_array.reshape(-1, img_size, img_size, 3)


def show_image(filepath):
	"""
	Displays the image from filepath. 		
	"""
	img = mpimg.imread(filepath)
	imgplot = plt.imshow(img)
	plt.show()


def plot_training(history_dict):
	"""
	Plots the training accuracy and loss graphs for a Keras model (validation graphs not included). 
	Parameters:
		- history_dict: a dictionary of "history" object from Keras 
	"""
	# plot training accuracy graph 
	plt.plot(history_dict['accuracy'])
	plt.title('Training Accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.show()

	# plot training loss graph 
	plt.plot(history_dict['loss'])
	plt.title('Training Loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.show() 


def plot_graphs(history_dict):
	"""
	Plots the training and validation accuracy and loss graphs for a Keras model (validation graphs included). 
	Parameters:
		- history_dict: a dictionary of "history" object from Keras 
	"""

	# plot training and validation accuracy graphs 
	plt.plot(history_dict['accuracy'])
	plt.plot(history_dict['val_accuracy'])
	plt.title('Training and Validation Accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()

	# plot training and validation loss graphs 
	plt.plot(history_dict['loss'])
	plt.plot(history_dict['val_loss'])
	plt.title('Training and Validation Loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()

