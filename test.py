import tensorflow as tf;
import buggy_car_data;


sess = tf.InteractiveSession();


#sess = tf.Session();

def _decode_img(filename, label):
	#print(type(filename));
	#print(dir(filename));
	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_image(image_string)
	#image_resized = tf.image.resize_images(image_decoded, [28, 28])
	return image_decoded, label


def _images_L_bmp(filename):
	filename = filename.replace('images/', 'images_L_bmp/');
	filename = filename.replace('.jpg', '.jpg.bmp');
	return filename;


def prepare_dataset():
	(trainX, trainY) = buggy_car_data.load_train_data()
	trainX = list(map(_images_L_bmp, trainX))
	filenames = tf.constant(trainX);
	labels = tf.constant(trainY);
	dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
	dataset = dataset.map(_decode_img)
	return dataset;


