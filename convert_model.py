import os
import json
import h5py
import numpy as np

def convert_h5_to_tfjs(h5_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load H5 and read weights
    print(f"Loading weights from {h5_path}...")
    weights_data = []
    manifest_weights = []
    
    # Define exact path to weights datasets inside H5 for the new 4-layer CNN architecture
    weight_mappings = [
        ("conv2d/kernel", "model_weights/conv2d/sequential/conv2d/kernel", [3, 3, 3, 32]),
        ("conv2d/bias", "model_weights/conv2d/sequential/conv2d/bias", [32]),
        ("conv2d_1/kernel", "model_weights/conv2d_1/sequential/conv2d_1/kernel", [3, 3, 32, 64]),
        ("conv2d_1/bias", "model_weights/conv2d_1/sequential/conv2d_1/bias", [64]),
        ("conv2d_2/kernel", "model_weights/conv2d_2/sequential/conv2d_2/kernel", [3, 3, 64, 128]),
        ("conv2d_2/bias", "model_weights/conv2d_2/sequential/conv2d_2/bias", [128]),
        ("conv2d_3/kernel", "model_weights/conv2d_3/sequential/conv2d_3/kernel", [3, 3, 128, 128]),
        ("conv2d_3/bias", "model_weights/conv2d_3/sequential/conv2d_3/bias", [128]),
        ("dense/kernel", "model_weights/dense/sequential/dense/kernel", [25088, 128]),
        ("dense/bias", "model_weights/dense/sequential/dense/bias", [128]),
        ("dense_1/kernel", "model_weights/dense_1/sequential/dense_1/kernel", [128, 1]),
        ("dense_1/bias", "model_weights/dense_1/sequential/dense_1/bias", [1])
    ]
    
    with h5py.File(h5_path, 'r') as f:
        binary_data = bytearray()
        
        for name, path, shape in weight_mappings:
            if path in f:
                dset = f[path][()]
                # Verify shape
                if list(dset.shape) != shape:
                    print(f"Warning: shape mismatch for {name}. Expected {shape}, got {list(dset.shape)}")
                
                # Convert to float32 little endian
                arr = dset.astype(np.float32)
                arr_bytes = arr.tobytes()
                
                print(f"Layer: {name}, Shape: {arr.shape}, Bytes: {len(arr_bytes)}")
                
                binary_data.extend(arr_bytes)
                manifest_weights.append({
                    "name": name,
                    "shape": shape,
                    "dtype": "float32"
                })
            else:
                raise KeyError(f"Weight path not found in H5 file: {path}. (Please train the new model using the updated notebook first to generate the new weights.)")
                
        # Write binary shard file
        shard_filename = "group1-shard1of1.bin"
        shard_path = os.path.join(output_dir, shard_filename)
        with open(shard_path, "wb") as bf:
            bf.write(binary_data)
        print(f"Successfully wrote weights to {shard_path} ({len(binary_data)} bytes)")
        
    # 2. Build model topology config (4-layer CNN with padding='same')
    model_config = {
        "class_name": "Sequential",
        "config": {
            "name": "sequential",
            "layers": [
                {
                    "class_name": "Conv2D",
                    "config": {
                        "name": "conv2d",
                        "trainable": True,
                        "dtype": "float32",
                        "batch_input_shape": [None, 224, 224, 3],
                        "filters": 32,
                        "kernel_size": [3, 3],
                        "strides": [1, 1],
                        "padding": "same",
                        "data_format": "channels_last",
                        "dilation_rate": [1, 1],
                        "groups": 1,
                        "activation": "relu",
                        "use_bias": True,
                        "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None
                    }
                },
                {
                    "class_name": "MaxPooling2D",
                    "config": {
                        "name": "max_pooling2d",
                        "trainable": True,
                        "dtype": "float32",
                        "pool_size": [2, 2],
                        "padding": "valid",
                        "strides": [2, 2],
                        "data_format": "channels_last"
                    }
                },
                {
                    "class_name": "Conv2D",
                    "config": {
                        "name": "conv2d_1",
                        "trainable": True,
                        "dtype": "float32",
                        "filters": 64,
                        "kernel_size": [3, 3],
                        "strides": [1, 1],
                        "padding": "same",
                        "data_format": "channels_last",
                        "dilation_rate": [1, 1],
                        "groups": 1,
                        "activation": "relu",
                        "use_bias": True,
                        "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None
                    }
                },
                {
                    "class_name": "MaxPooling2D",
                    "config": {
                        "name": "max_pooling2d_1",
                        "trainable": True,
                        "dtype": "float32",
                        "pool_size": [2, 2],
                        "padding": "valid",
                        "strides": [2, 2],
                        "data_format": "channels_last"
                    }
                },
                {
                    "class_name": "Conv2D",
                    "config": {
                        "name": "conv2d_2",
                        "trainable": True,
                        "dtype": "float32",
                        "filters": 128,
                        "kernel_size": [3, 3],
                        "strides": [1, 1],
                        "padding": "same",
                        "data_format": "channels_last",
                        "dilation_rate": [1, 1],
                        "groups": 1,
                        "activation": "relu",
                        "use_bias": True,
                        "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None
                    }
                },
                {
                    "class_name": "MaxPooling2D",
                    "config": {
                        "name": "max_pooling2d_2",
                        "trainable": True,
                        "dtype": "float32",
                        "pool_size": [2, 2],
                        "padding": "valid",
                        "strides": [2, 2],
                        "data_format": "channels_last"
                    }
                },
                {
                    "class_name": "Conv2D",
                    "config": {
                        "name": "conv2d_3",
                        "trainable": True,
                        "dtype": "float32",
                        "filters": 128,
                        "kernel_size": [3, 3],
                        "strides": [1, 1],
                        "padding": "same",
                        "data_format": "channels_last",
                        "dilation_rate": [1, 1],
                        "groups": 1,
                        "activation": "relu",
                        "use_bias": True,
                        "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None
                    }
                },
                {
                    "class_name": "MaxPooling2D",
                    "config": {
                        "name": "max_pooling2d_3",
                        "trainable": True,
                        "dtype": "float32",
                        "pool_size": [2, 2],
                        "padding": "valid",
                        "strides": [2, 2],
                        "data_format": "channels_last"
                    }
                },
                {
                    "class_name": "Flatten",
                    "config": {
                        "name": "flatten",
                        "trainable": True,
                        "dtype": "float32",
                        "data_format": "channels_last"
                    }
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "dense",
                        "trainable": True,
                        "dtype": "float32",
                        "units": 128,
                        "activation": "relu",
                        "use_bias": True,
                        "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None
                    }
                },
                {
                    "class_name": "Dropout",
                    "config": {
                        "name": "dropout",
                        "trainable": True,
                        "dtype": "float32",
                        "rate": 0.5,
                        "seed": None,
                        "noise_shape": None
                    }
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "dense_1",
                        "trainable": True,
                        "dtype": "float32",
                        "units": 1,
                        "activation": "sigmoid",
                        "use_bias": True,
                        "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None
                    }
                }
            ]
        }
    }
    
    # 3. Create model.json structure
    model_json = {
        "format": "layers-model",
        "generatedBy": "keras-converter",
        "convertedBy": "Custom H5-to-TFJS converter script",
        "modelTopology": {
            "keras_version": "2.15.0",
            "backend": "tensorflow",
            "model_config": model_config
        },
        "weightsManifest": [
            {
                "paths": [shard_filename],
                "weights": manifest_weights
            }
        ]
    }
    
    # Write model.json
    model_json_path = os.path.join(output_dir, "model.json")
    with open(model_json_path, "w") as jf:
        json.dump(model_json, jf, indent=2)
    print(f"Successfully wrote model configuration to {model_json_path}")

if __name__ == "__main__":
    h5_path = r"c:\DATA FAKHRI\TUGAS\Semester8\ComputerVision\FaceDetectionWeb\face_mask_cnn.h5"
    output_dir = r"c:\DATA FAKHRI\TUGAS\Semester8\ComputerVision\FaceDetectionWeb\web\model"
    # Note: Running this will raise an error until the user trains the new model to generate the updated face_mask_cnn.h5
    try:
        convert_h5_to_tfjs(h5_path, output_dir)
    except Exception as e:
        print(f"Conversion skipped for now: {e}")
