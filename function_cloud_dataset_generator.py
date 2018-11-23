import io
import numpy as np
from function_cloud_maker import generate_function_cloud
import tensorflow as tf


def _int64_feature(int_list):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=int_list))


def make_function_cloud_dataset(number_of_iterations, function_list):
	counter = 0

	FILE1 = open('function_cloud_dataset_1.txt', 'w')
	FILE2 = open('function_cloud_label_values_1.txt', 'w')

	FILE1_string = ""
	FILE2_list   = []

	#for tf record
	# tfrecords_filename = 'flattened_function_cloud_only_one.tfrecords'
	# writer = tf.python_io.TFRecordWriter(tfrecords_filename)

	function_cloud_dataset = []

	function_cloud_set, function_true_values = map(list, zip(*[generate_function_cloud(np.random.choice(function_list), random_start=True, return_type=True) for _ in range(number_of_iterations)]))

	for data_cloud, label in zip(function_cloud_set, np.array(function_true_values)):
	
		data_strip = [[1 if i in [j for j, x in enumerate([round(_*5.0)/5.0 for _ in data_cloud]) if x == round(layer_value*5.0)/5.0] else 0 for i in range(101)] for layer_value in np.arange(-10,10+0.2, 0.2)]
		data_strip = np.array(list(sum(data_strip, [])))
		# data_strip = np.array2string(data_strip, separator=',', formatter={'int':lambda x: "%d" %x})
		FILE1_string = FILE1_string + str(data_strip).replace("\n", "") + "\n"
		FILE2_list.append(label)
		# print FILE1_string

		#for tf record
		# example = tf.train.Example(features = tf.train.Features(feature = {
		# 	'image_label': _int64_feature([label]),
		# 	'image_string': _int64_feature(data_strip)}))

		# writer.write(example.SerializeToString())
		
		counter += 1

		print counter

	FILE1.write(FILE1_string)
	FILE2_list = np.array2string(np.array(FILE2_list), separator=',', formatter={'int':lambda x: "%d" %x})
	FILE2.write(FILE2_list)

	FILE1.close()
	FILE2.close()

	# writer.close()

	return function_cloud_dataset, function_true_values


if __name__ == '__main__':

	np.set_printoptions(threshold=np.nan)

	number_of_iterations = 10000

	function_list = ['linear', 'quadratic', 'exponential', 'simple_periodic']

	function_cloud_dataset, function_true_values = make_function_cloud_dataset(number_of_iterations, function_list)

	# text_file1 = open("function_cloud_dataset_1.txt", "w")
	# for flattened_image in function_cloud_dataset:
	# 	text_file1.write("%s\n" % str(flattened_image))
	# text_file1.close()

	# text_file2 = open("function_true_values_1.txt", "w")
	# text_file2.write("%s" % function_true_values)
	# text_file2.close()