import tensorflow as tf
import os.path
from model import Model
import sys

FLAGS = tf.app.flags.FLAGS
#NUM_LABELS = 4
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
NUMBER_OF_CHANNELS = 3
#SOURCE_DIR = './data/'
#TRAINING_IMAGES_DIR = SOURCE_DIR + 'train/'
#LIST_FILE_NAME = 'list.txt'
BATCH_SIZE = 1
#TRAINING_SET_SIZE = 81153
#TRAIN_FILE = "/home/sebv/SebV/datas/testTF2/*/*.png"
#TRAIN_FILE = "'C:/Users/Neogeoisie/Desktop/projetTer/55_espece_no_part"
TRAIN_FILE = "train_data/*/*.png"

def creation_batch(filename_queue):	
	print("START")
	image_reader = tf.WholeFileReader()
	key, image_file = image_reader.read(filename_queue)
	S = tf.string_split([key],'/')
	length = tf.cast(S.dense_shape[1],tf.int32)
	# adjust constant value corresponding to your paths if you face issues. It should work for above format.
	label = S.values[length-tf.constant(2,dtype=tf.int32)]
	label = tf.string_to_number(label,out_type=tf.int32)	
	image = tf.image.decode_png(image_file,3,tf.uint16)
	image = tf.cast(image, tf.float32)
	image = tf.image.resize_images(image,[IMAGE_WIDTH,IMAGE_HEIGHT])
	image = tf.reshape(image, [IMAGE_WIDTH,IMAGE_HEIGHT,NUMBER_OF_CHANNELS])
	label = tf.cast(label, tf.int64)
	images_batch, labels_batch = tf.train.shuffle_batch([image,label],batch_size=BATCH_SIZE,capacity=5000,min_after_dequeue=1000)
	labels_batch = tf.reshape(labels_batch, [BATCH_SIZE])
	return images_batch, labels_batch

def train(train_file):
	model = Model()
	with tf.Graph().as_default():
		x = tf.placeholder(tf.float32, [None,IMAGE_WIDTH,IMAGE_HEIGHT,NUMBER_OF_CHANNELS], name='x-input')
		y = tf.placeholder(tf.int32, [None], name='y-input')
		train_file =str(train_file)
		print ("THISSS : " + train_file)
		#print type(train_file)
		#print (TRAIN_FILE)
		#print type(TRAIN_FILE)
		#os.system("find -type f -name " +TRAIN_FILE +" | wc -l")
		#filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(train_file))
		filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(train_file))
		images_batch, labels_batch = creation_batch(filename_queue)
		#images_batch = creation_batch(filename_queue)
		# Start a new session to show example output.
		keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
		global_step = tf.train.get_or_create_global_step()
		logits = model.inference(images_batch, keep_prob=keep_prob) #fonction de prediction
		loss = model.loss(logits=logits, labels=labels_batch)
		accuracy = model.accuracy(logits, labels_batch)
		summary_op = tf.summary.merge_all()
		train_op = model.train(loss, global_step=global_step)
		graph=tf.get_default_graph()
		predictions=model.predictions(logits=logits)
		top_k_op = tf.nn.in_top_k(logits, labels_batch, 1)
		reader = tf.TextLineReader()
		key, value = reader.read(filename_queue)
		#print key,value
		saver = tf.train.Saver()
		
		with tf.Session() as sess:
			# Required to get the filename matching to run.
			writer = tf.summary.FileWriter("graph", sess.graph)
			sess.run(tf.local_variables_initializer())
			sess.run(tf.global_variables_initializer())
			# Coordinate the loading of image files.
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			for i in range(10):
				# Get an image tensor and print its value.
				img=sess.run([images_batch])
				print("zzz")
				im_batch, lab_batch = sess.run([images_batch, labels_batch])
				_, cur_loss, summary = sess.run([train_op, loss, summary_op],feed_dict={x: im_batch, y: lab_batch, keep_prob: 0.5})
				curr=sess.run(accuracy,feed_dict={x: im_batch, y: lab_batch, keep_prob: 0.5})
				r=sess.run(predictions,feed_dict={x: im_batch, y: lab_batch, keep_prob: 0.5})
				z=sess.run(logits,feed_dict={x: im_batch, y: lab_batch, keep_prob: 0.5})
				if i%10==0:
					print("iteration "+str(i))
					print (curr)
				writer.add_summary(summary, i)
				save_path=saver.save(sess,"C:/Users/Neogeoisie/Desktop/projetTer/datas/tfRecording/model.ckpt")
				print("Model saved in file: %s" % save_path)
			coord.request_stop()
			coord.join(threads)
		
def main(useless):
	#dossier_source = sys.argv[1] #dossier contenant les dossiers poissons   
	#files= dossier_source +"*/*.png"
	train(TRAIN_FILE)

if __name__ == '__main__':
	tf.app.flags.DEFINE_integer('batch_size', 1, 'size of training batches')
	tf.app.flags.DEFINE_integer('num_iter', 2, 'number of training iterations')
	tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt-10000', 'path to checkpoint file')
	tf.app.flags.DEFINE_string('train_data', 'data', 'path to train and test data')
	tf.app.flags.DEFINE_string('summary_dir', 'graphs', 'path to directory for storing summaries')
	tf.app.run()