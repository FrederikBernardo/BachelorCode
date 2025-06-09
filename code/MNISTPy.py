import numpy as np
import struct

def read_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}")
        
        images = np.frombuffer(f.read(), dtype=np.uint8)

        images = images.reshape(num_images, num_rows, num_cols)
        
        return images

def read_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic}.")
        
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return labels

