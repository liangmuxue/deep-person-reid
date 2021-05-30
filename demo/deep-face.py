from deepface import DeepFace
from deepface.extendedmodels import Age, Gender, Race, Emotion

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.python.keras import backend as keras

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
keras.set_session(tf.compat.v1.Session(config=config))

models = {}
models["emotion"] = Emotion.loadModel()
models["age"] = Age.loadModel()
models["gender"] = Gender.loadModel()
models["race"] = Race.loadModel()
ret = DeepFace.analyze("/home/bavon/face_test/capture_lmx_cgq2.jpg", models=models)

print("Age: ", ret["age"])
print("Gender: ", ret["gender"])

