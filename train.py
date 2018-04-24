import tensorflow as tf;
import numpy as np;
import buggy_car_data;

batch_size = 10

def cnn_model_fn(features, labels, mode):
  #"""Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # buggy_car_data images are 128x96 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, 128, 96, 1])
  #input_layer = tf.reshape(features, [batch_size, 128, 96, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 128, 96, 1]
  # Output Tensor Shape: [batch_size, 128, 96, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 128, 96, 32]
  # Output Tensor Shape: [batch_size, 64, 48, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 64, 48, 32]
  # Output Tensor Shape: [batch_size, 64, 48, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 64, 48, 64]
  # Output Tensor Shape: [batch_size, 32, 24, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #3
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 24, 64]
  # Output Tensor Shape: [batch_size, 32, 24, 128]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 32, 24, 128]
  # Output Tensor Shape: [batch_size, 16, 12, 128]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Convolutional Layer #4
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 16, 12, 128]
  # Output Tensor Shape: [batch_size, 16, 12, 256]
  conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=256,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #4
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 16, 12, 256]
  # Output Tensor Shape: [batch_size, 8, 6, 256]
  pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

  # Convolutional Layer #5
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 8, 6, 256]
  # Output Tensor Shape: [batch_size, 8, 6, 512]
  conv5 = tf.layers.conv2d(
      inputs=pool4,
      filters=512,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #5
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 8, 6, 512]
  # Output Tensor Shape: [batch_size, 4, 3, 512]
  pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 4, 3, 512]
  # Output Tensor Shape: [batch_size, 4 * 3 * 512]
  pool5_flat = tf.reshape(pool5, [-1, 4 * 3 * 512])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 4 * 3 * 512]
  # Output Tensor Shape: [batch_size, 2048]
  dense = tf.layers.dense(inputs=pool5_flat, units=2048, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 2048]
  # Output Tensor Shape: [batch_size, 21]
  logits = tf.layers.dense(inputs=dropout, units=21)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    print('return PREDICT');
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=[batch_size])
 
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  print('return EstimatorSpec');
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
	# Load training and eval data
	train_input_fn = buggy_car_data.train_input_fn(batch_size);
	eval_input_fn = buggy_car_data.test_input_fn(batch_size);

	# Create the Estimator
	buggy_car_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn,
		model_dir="./models")

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log,
		every_n_iter=50)

	# Train the model
	buggy_car_classifier.train(
		#input_fn=train_input_fn,
		input_fn= lambda:buggy_car_data.train_input_fn(batch_size),
		#steps=20000,
		steps=1,
		hooks=[logging_hook])

	# Evaluate the model and print results
	eval_results = buggy_car_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)




if __name__ == "__main__":
	#config = tf.ConfigProto(allow_soft_placement=True);
	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5);
	#config.gpu_options.allow_growth=True;
	
	tf.app.run(main)
  
