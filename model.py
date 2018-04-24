import tensorflow as tf


class Model(object):
	def __init__(self, batch_size=32, learning_rate=1e-4, num_labels=15):
		self._batch_size = batch_size
		self._learning_rate = learning_rate
		self._num_labels = num_labels
	
	def inference(self, images, keep_prob):
		with tf.name_scope('input'):
			x = tf.reshape(images, [-1, 64, 64, 3])
			self.activation_summary(x)
			
		with tf.variable_scope('conv1') as scope:
			kernel = self.weights([5, 5, 3, 32])
			conv = self.conv(x, kernel)
			bias = self.bias([32])
			preactivation = tf.nn.bias_add(conv, bias)
			conv1 = tf.nn.relu(preactivation, name=scope.name)
			print (conv1.get_shape())
			self.activation_summary(conv1)
			with tf.variable_scope('visualization'):
				# scale weights to [0 1], type is still float
				x_min = tf.reduce_min(kernel)
				x_max = tf.reduce_max(kernel)
				kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
				# to tf.image_summary format [batch_size, height, width, channels]
				kernel_transposed = tf.transpose (kernel_0_to_1, [3, 0, 1, 2])
				# this will display random 3 filters from the 64 in conv1
				tf.summary.image('conv1/filters', kernel_transposed, max_outputs=64)
		
		with tf.variable_scope('conv2') as scope:
			kernel = self.weights([5, 5, 32, 64])
			conv = self.conv(conv1, kernel)
			bias = self.bias([64])
			preactivation = tf.nn.bias_add(conv, bias)
			conv2 = tf.nn.relu(preactivation, name=scope.name)
			print (conv2.get_shape())
			self.activation_summary(conv2)

		with tf.variable_scope('conv3') as scope:
			kernel = self.weights([3, 3, 32, 128])
			conv = self.conv(conv1, kernel)
			bias = self.bias([128])
			preactivation = tf.nn.bias_add(conv, bias)
			conv3 = tf.nn.relu(preactivation, name=scope.name)
			print (conv3.get_shape())
			self.activation_summary(conv3)

		with tf.variable_scope('local1') as scope:
			reshape = tf.reshape(conv3, [-1, 64 * 64 * 128])
			W_fc1 = self.weights([64 * 64 * 128, 1024])
			b_fc1 = self.bias([1024])
			local1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1, name=scope.name)
			self.activation_summary(local1)
		
		with tf.variable_scope('local2_linear') as scope:
			W_fc2 = self.weights([1024, self._num_labels])
			b_fc2 = self.bias([self._num_labels])
			local1_drop = tf.nn.dropout(local1, keep_prob)
			local2 = tf.nn.bias_add(tf.matmul(local1_drop, W_fc2), b_fc2, name=scope.name)
			self.activation_summary(local2)
		return local2
	
	def train(self, loss, global_step):
		tf.summary.scalar('learning_rate', self._learning_rate)
		train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss, global_step=global_step)
		return train_op
	
	def loss(self, logits, labels):
		with tf.variable_scope('loss') as scope:
			print (logits.get_shape())
			cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
			print (cross_entropy.get_shape())
			cost = tf.reduce_mean(cross_entropy, name=scope.name)
			print (cost.get_shape())
			tf.summary.scalar('cost', cost)
		return cost
		
	def predictions(self, logits):
		with tf.variable_scope('predictions') as scope:
			predictions=tf.nn.softmax(logits, name='pred')
			tf.summary.scalar('predictions', predictions)
		return predictions
		
	def accuracy(self, logits, y):
		with tf.variable_scope('accuracy') as scope:
			accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits, 1), dtype=tf.int64), y), dtype=tf.float32),name=scope.name)
			tf.summary.scalar('accuracy', accuracy)
		return accuracy
		
	def conv(self, x, W):
		return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')
			
	def atrous_conv(self, x, W, rate):
		return tf.nn.conv2d(input=x, filter=W, rate=rate, padding='SAME')
		
	def max_pool(self, input, shape, stride):
		return tf.nn.max_pool(value=input, ksize=[1, shape, shape, 1], strides=[1, stride, stride, 1], padding='SAME')
			
	def avg_pool(self, shape, stride):
		return tf.nn.avg_pool(value=input, ksize=[1, shape, shape, 1], strides=[1, stride, stride, 1], padding='SAME')
		
	def batch_norm(self, x):
		return tf.nn.batch_normalization(value=input)
		
	def weights(self, shape):
		return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32), name='weights')
		
	def bias(self, shape):
		return tf.Variable(tf.constant(1., shape=shape, dtype=tf.float32), name='bias')
		
	def activation_summary(self, var):
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)