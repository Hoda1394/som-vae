"""
Utility functions for the SOM-VAE model
Copyright (c) 2018
Author: Vincent Fortuin
Institution: Biomedical Informatics group, ETH Zurich
License: MIT License
"""

from pathlib import Path
import nibabel as nib
import tensorflow as tf
import numpy as np
import math
import glob
import time


def interpolate_arrays(arr1, arr2, num_steps=100, interpolation_length=0.3):
    """Interpolates linearly between two arrays over a given number of steps.
    The actual interpolation happens only across a fraction of those steps.

    Args:
        arr1 (np.array): The starting array for the interpolation.
        arr2 (np.array): The end array for the interpolation.
        num_steps (int): The length of the interpolation array along the newly created axis (default: 100).
        interpolation_length (float): The fraction of the steps across which the actual interpolation happens (default: 0.3).

    Returns:
        np.array: The final interpolated array of shape ([num_steps] + arr1.shape).
    """
    assert arr1.shape == arr2.shape, "The two arrays have to be of the same shape"
    start_steps = int(num_steps*interpolation_length)
    inter_steps = int(num_steps*((1-interpolation_length)/2))
    end_steps = num_steps - start_steps - inter_steps
    interpolation = np.zeros([inter_steps]+list(arr1.shape))
    arr_diff = arr2 - arr1
    for i in range(inter_steps):
        interpolation[i] = arr1 + (i/(inter_steps-1))*arr_diff
    start_arrays = np.concatenate([np.expand_dims(arr1, 0)] * start_steps)
    end_arrays = np.concatenate([np.expand_dims(arr2, 0)] * end_steps)
    final_array = np.concatenate((start_arrays, interpolation, end_arrays))
    return final_array


def compute_NMI(cluster_assignments, class_assignments):
    """Computes the Normalized Mutual Information between cluster and class assignments.
    Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    
    Args:
        cluster_assignments (list): List of cluster assignments for every point.
        class_assignments (list): List of class assignments for every point.

    Returns:
        float: The NMI value.
    """
    assert len(cluster_assignments) == len(class_assignments), "The inputs have to be of the same length."
    
    clusters = np.unique(cluster_assignments)
    classes = np.unique(class_assignments)

    #print(clusters,classes)
    
    num_samples = len(cluster_assignments)
    num_clusters = len(clusters)
    num_classes = len(classes)
    
    assert num_classes > 1, "There should be more than one class."
        
    cluster_class_counts = {cluster_: {class_: 0 for class_ in classes} for cluster_ in clusters}

    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_.numpy()][class_] += 1
    
    cluster_sizes = {cluster_: sum(list(class_dict.values())) for cluster_, class_dict in cluster_class_counts.items()}
    class_sizes = {class_: sum([cluster_class_counts[clus][class_] for clus in clusters]) for class_ in classes}
    
    I_cluster_class = H_cluster = H_class = 0
    
    for cluster_ in clusters:
        for class_ in classes:
            if cluster_class_counts[cluster_][class_] == 0:
                pass
            else:
                I_cluster_class += (cluster_class_counts[cluster_][class_]/num_samples) * \
                (np.log((cluster_class_counts[cluster_][class_]*num_samples)/ \
                        (cluster_sizes[cluster_]*class_sizes[class_])))
                        
    for cluster_ in clusters:
        H_cluster -= (cluster_sizes[cluster_]/num_samples) * np.log(cluster_sizes[cluster_]/num_samples)
                
    for class_ in classes:
        H_class -= (class_sizes[class_]/num_samples) * np.log(class_sizes[class_]/num_samples)
        
    NMI = (2*I_cluster_class)/(H_cluster+H_class)
    
    return NMI


def compute_purity(cluster_assignments, class_assignments):
    """Computes the purity between cluster and class assignments.
    Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    
    Args:
        cluster_assignments (list): List of cluster assignments for every point.
        class_assignments (list): List of class assignments for every point.

    Returns:
        float: The purity value.
    """
    assert len(cluster_assignments) == len(class_assignments)
    
    num_samples = len(cluster_assignments)
    num_clusters = len(np.unique(cluster_assignments))
    num_classes = len(np.unique(class_assignments))
    
    cluster_class_counts = {cluster_: {class_: 0 for class_ in np.unique(class_assignments)}
                            for cluster_ in np.unique(cluster_assignments)}
    
    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_.numpy()][class_] += 1
        
    total_intersection = sum([max(list(class_dict.values())) for cluster_, class_dict in cluster_class_counts.items()])
    
    purity = total_intersection/num_samples
    
    return purity

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def serialize_example(img, shape):
    feature = {
        'img' : _bytes_feature(img),
        'shape' : _int64_feature(shape),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def parse_2d_image(record):
    image_feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'shape': tf.io.FixedLenFeature([2], tf.int64)
    }
    data = tf.io.parse_single_example(record, image_feature_description)
    img = data['img']
    img = tf.io.decode_raw(img, tf.uint8)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, data['shape'])
    #img = tf.image.resize(img, (2**target_res, 2**target_res))
    #img = adjust_dynamic_range(img, [0.0, 255.0], [-1.0, 1.0])
    return img

def write_cifti_tfrecords(data_pattern,tfrecords_folder,size_shard=50,compressed=False):

    # Data folder
    img_filenames = list(glob.glob(data_pattern))
    assert img_filenames, 'No files found'

    # TF records folder
    tfrecords_folder = Path(tfrecords_folder)
    tfrecords_folder.mkdir(parents=True, exist_ok=True)

    # Shard parameters
    num_samples=len(img_filenames)
    print('Number of samples found: {}'.format(num_samples))
    num_shards = math.ceil(num_samples/size_shard)

    shards = np.array_split(img_filenames,num_shards)   # PB ?
    tfrecords_filename = []
    progbar = tf.keras.utils.Progbar(target=num_samples, verbose=True)

    for i in range(num_shards):
        cifti_paths = shards[i]
        tfrecords_filename = str(tfrecords_folder.joinpath('tfrecords_train{}.tfrecord'.format(i)))
        #if not compressed: tfrecords_writer = tf.io.TFRecordWriter(tfrecords_filename,options=None)
        #elif compressed: tfrecords_writer = tf.io.TFRecordWriter(tfrecords_filename,options=tf.io.TFRecordOptions(compression_type='GZIP'))
        with tf.io.TFRecordWriter(tfrecords_filename) as tfrecords_writer:
            for cifti_path in cifti_paths:
                sample_data=nib.load(cifti_path).get_fdata()
                sample_data=255*(sample_data-sample_data.min())/(sample_data.max()-sample_data.min())
                sample_shape=np.array(sample_data.shape).astype(np.int64)
                sample_data_raveled = sample_data.astype(np.uint8).ravel().tostring()
        
                tfrecords_writer.write(serialize_example(sample_data_raveled,sample_shape))
                progbar.add(1)
        

def adjust_range(sample):
    sample = (sample - tf.reduce_min(sample))/(tf.reduce_max(sample)-tf.reduce_min(sample))
    return sample

def epoch(sample,batch_size):

    #if sample.shape[0]%batch_size != 0: print('Batch size does not suit scan duration, excess data will be discarded')
    #sample = tf.convert_to_tensor(sample)
    #print(sample.shape,tf.shape(sample)[1])
    #series_shape = tf.shape(sample)
    #block_shape = np.asarray([batch_size,series_shape[1]],dtype=np.int32)
    #num_blocks = series_shape // block_shape
    sample = tf.reshape(sample,[409,2,65890])
    #sample = tf.reshape(sample, np.insert(block_shape,0,num_blocks[0]))
    return sample

def get_dataset(tfrecords_folder,epoch_size,batch_size):
    # Did not standardize, did adjust the range to 0-1, prefetch might affect memory
    with tf.device('cpu:0'):
        tfrecords_folder = Path(tfrecords_folder)
        assert tfrecords_folder.is_dir(), 'No tfrecords folder to process'

        file_pattern = glob.glob(str(tfrecords_folder.joinpath("*.tfrecord")))
        assert file_pattern, 'No files in folder'

        dataset = tf.data.Dataset.list_files(str(tfrecords_folder.joinpath("*.tfrecord")))
        dataset = dataset.interleave(lambda x: 
            tf.data.TFRecordDataset(x, compression_type=None),
            cycle_length=1,block_length=4)
        print(tf.executing_eagerly())
        dataset = dataset.map(lambda x: parse_2d_image(x),num_parallel_calls=1)
        dataset = dataset.shuffle(buffer_size=20)

        #dataset = dataset.map(lambda x: tf.py_function(epoch,[x,epoch_size],[]))
        dataset = dataset.map(lambda x: epoch(x,epoch_size))
        print(len(list(dataset.as_numpy_iterator())))
        dataset = dataset.unbatch()
        print(len(list(dataset.as_numpy_iterator())))
        dataset = dataset.batch(batch_size,drop_remainder=True)
        #dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset




            




