import os
import random
import numpy as np
import tensorflow as tf

def set_global_seed(seed: int = 42) -> None:
    # Set TensorFlow parallelism settings before importing TensorFlow
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize TensorFlow's random seed
    tf.random.set_seed(seed)

    # Ensure reproducibility with TensorFlow's operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
