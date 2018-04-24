import os
import pandas as pd
import tensorflow as tf;
csvColumnNames = ['imageFolder', 'imageName', 'throOUT', 'rudoOUT']

def load_train_data():
	trainPath = os.getcwd()+"/train-data";
	trainCSVs = trainPath + '/csv';
	trainX = [];
	trainY = [];
	tempX = [];

	## train data
	for root, dirs, files in os.walk(trainCSVs):
		for file in files:
			csv = pd.read_csv(trainCSVs+'/'+file, names=csvColumnNames);
			tempX += list(csv['imageFolder']+'/'+csv['imageName']);
			trainY += list(csv['rudoOUT']);

	for idx, imgPath in enumerate(tempX):
		a1 = os.getcwd() + '/train-data/' +os.path.normpath(imgPath);
		a1 = a1.replace('/home/pi/myProject/python/rc-tracked-car/', '')
		# a1 = a1.replace('images/', 'images_L_bmp/')
		# "/home/pi/myProject/python/rc-tracked-car/images/2018-03-21-17.10.25","img_2018-03-21-17.10.27.652620.jpg",49,360

		if not os.path.isfile(a1):
		# if not os.path.exists(a1):
			print(idx, a1 , 'not exist');
			print(os.path.normpath(imgPath));
			break;
		trainX.append(a1);
	# print(len(trainX))
	# print(len(trainY))
	
	return (trainX, trainY);

def load_test_data():
	testPath = os.getcwd()+"/test-data";
	testCSVs = testPath + '/csv';
	testX = [];
	testY = [];
	tempX = [];

	## train data
	for root, dirs, files in os.walk(testCSVs):
		for file in files:
			csv = pd.read_csv(testCSVs+'/'+file, names=csvColumnNames);
			tempX += list(csv['imageFolder']+'/'+csv['imageName']);
			testY += list(csv['rudoOUT']);

	for idx, imgPath in enumerate(tempX):
		a1 = os.getcwd() + '/test-data/' +os.path.normpath(imgPath);
		a1 = a1.replace('/home/pi/myProject/python/rc-tracked-car/', '')
		# a1 = a1.replace('images/', 'images_L_bmp/')

		if not os.path.isfile(a1):
		# if not os.path.exists(a1):
			print(idx, a1 , 'not exist');
			print(os.path.normpath(imgPath));
			break;
		testX.append(a1);
	# print(len(testX))
	# print(len(testY))
	
	return (testX, testY);

def get_train_csv():
	trainPath = os.getcwd()+"/train-data";
	trainCSVs = trainPath + '/csv';

	returnFiles = [];
	for root, dirs, files in os.walk(trainCSVs):
		for file in files:
			returnFiles.append(trainCSVs + '/' + file);
	
	return returnFiles;

def get_test_csv():
	testPath = os.getcwd()+"/test-data";
	testCSVs = testPath + '/csv';

	returnFiles = [];
	for root, dirs, files in os.walk(testCSVs):
		for file in files:
			returnFiles.append(testCSVs + '/' + file);
	
	return returnFiles;


def _decode_img(filename, label):
	#print(type(filename));
	#print(dir(filename));
	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_bmp(
		image_string,
		channels=0,
		name='decode_bmp')
	image_decoded.set_shape([128, 96, 1]);
	image_decoded = tf.cast(image_decoded, tf.float32, name='image')
	#print(image_decoded);
	#image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 128, 96)
	#image_resized = tf.image.resize_images(image_decoded, [128, 96])
	return image_decoded, label

def _convert_label(label):
	return int(float(label)/36. + 10)

def _images_L_bmp(filename):
	filename = filename.replace('images/', 'images_L_bmp/');
	filename = filename.replace('.jpg', '.jpg.bmp');
	return filename;

def features():
	(trainX, trainY) = load_train_data();
	return set(trainY);

def train_dataset():
	(trainX, trainY) = load_train_data()
	trainX = list(map(_images_L_bmp, trainX))
	#trainY = list(map(_convert_label, trainY))
	filenames = tf.constant(trainX, name='filenames');
	labels = tf.constant(trainY, name='labels');
	dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
	dataset = dataset.map(_decode_img)
	return dataset;

def test_dataset():
	(testX, testY) = load_test_data()
	testX = list(map(_images_L_bmp, testX))
	#testY = list(map(_convert_label, testY))
	filenames = tf.constant(testX, name='filenames');
	labels = tf.constant(testY, name='labels');
	dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
	dataset = dataset.map(_decode_img)
	return dataset;


def train_input_fn(batch_size):
	"""An input function for training"""
	dataset = train_dataset();
	dataset = dataset.shuffle(batch_size).repeat().batch(batch_size);
	return dataset

def test_input_fn(batch_size):
	"""An input function for evaluation or prediction"""
	dataset = test_dataset();
	dataset = dataset.shuffle(batch_size).batch(batch_size);
	return dataset

if __name__ == '__main__':
	l1 = list(features());
	l1.sort();
	print('\ntrain data features:\n\t', l1);
	
	l2 = list(map(_convert_label, l1));
	print('\nconvert to 0 to 20:\n\t', l2);
	
	
	print();
	(trainX, trainY) = load_train_data();
	(testX, testY) = load_test_data();
	for value in trainY:
		if (type(value) != type(123)) :
			print(value)
			break;
	print('In this time, I sure every "trainY" value is type int');
	
	for v in testY:
		if (type(v) != type(123)) :
			print(v)
	print('In this time, I sure every "testY" value is type int');
		
