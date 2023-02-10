from config import CFG
import data_setup
import model_builder

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import AUC
from tensorflow_addons.metrics import F1Score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vectorize',
                    help="If True, vectorize the data. Otherwise load vectorized data",
                    default=True,
                    choices=('True', 'False'))
args = parser.parse_args()

# Load model
timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print(f"[INFO] ({timestamp}) Load SBERT Model...")
start = time.time()
s_bert = model_builder.load_sbert(CFG.SBERT_MODEL_PATH, CFG.SBERT_MODEL_FOLDER)
timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print(f"[INFO] ({timestamp}) Done! ({time.time()-start:.3f}s)")

# Encode text to vector
start = time.time()

data_path = Path(CFG.DATA_PATH)
vectorized_file = data_path / f"{CFG.VECTORIZED_FILE}"

if args.vectorize == True:
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    print(f"[INFO] ({timestamp}) Vectorize text data...")
    data_setup.save_encoded(data_path=CFG.DATA_PATH, 
                            file_name=CFG.MODEL_CSV_FILE,
                            save_name=CFG.VECTORIZED_FILE,
                            random_state=CFG.RANDOM_STATE,
                            s_bert=s_bert)
else:
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    print(f"[INFO] ({timestamp}) Load Vectorized Data...")

vectorized = np.load(f"{vectorized_file}")
X_train_vectorized = vectorized['X_train_vectorized']
y_train = vectorized['y_train']
X_test_vectorized = vectorized['X_test_vectorized']
y_test = vectorized['y_test']

timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print(f"[INFO] ({timestamp}) Done! ({time.time()-start:.3f}s)")

# Calculate class weights because of imbalanced data
class_weights = (len(y_train) - y_train.sum(axis=0)) / y_train.sum(axis=0)
class_weights = dict(enumerate(class_weights))


f1_score = F1Score(num_classes=CFG.N_CLASSES, average='weighted', name='F1_score')
prauc = AUC(curve='PR', multi_label=True, num_labels=CFG.N_CLASSES, name='PRAUC')

model = model_builder.get_classifier()
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[f1_score, prauc])


model_name = CFG.CLASSIFIER_MODEL_FILE
now = datetime.datetime.now().strftime("%y-%m-%d_%H%M")
save_path = f"{CFG.CLASSIFIER_MODEL_PATH}/{now}"

modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    f"{save_path}/{model_name}.h5",
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True
)
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=f'{save_path}/runs/',
    write_graph=True,
    update_freq='epoch'
)
timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print(f"[INFO] ({timestamp}) Model Training...")
history = model.fit(X_train_vectorized, y_train, 
                    epochs=CFG.EPOCHS, 
                    batch_size=CFG.BATCH_SIZE, 
                    class_weight=class_weights,
                    validation_data=(X_test_vectorized, y_test),
                    callbacks=[modelckpt_callback,
                               earlystop_callback,
                               tensorboard_callback])

timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print(f"[INFO] ({timestamp}) Done!")
print(f"[INFO] ({timestamp}) Model saved '{save_path}/{model_name}.h5'")