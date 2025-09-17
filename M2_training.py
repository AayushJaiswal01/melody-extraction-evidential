import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Conv2D, MaxPooling2D, Dense, Dropout, Reshape, TimeDistributed, add, BatchNormalization
from tensorflow.keras import Model
from keras.regularizers import l2
from glob import glob
from tqdm import tqdm
import mir_eval.melody
import keras
import sys

# --- GPU and General Configurations ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# --- PATHS ---
DATA_DIR = '../../../../../spl_v2_mir1k/npy_data'
BASE_MODEL_WEIGHTS_PATH = './model_weights/BASELINE_CLASSIFICATION_MODEL/' 
CONFIDENCE_MODEL_WEIGHTS_PATH = './model_weights/CONFIDENCE_MODEL/'
os.makedirs(BASE_MODEL_WEIGHTS_PATH, exist_ok=True)
os.makedirs(CONFIDENCE_MODEL_WEIGHTS_PATH, exist_ok=True)

# --- Data Loading ---
train_audio_files = sorted(glob(os.path.join(DATA_DIR, 'train/audio/*.npy')))
train_pitch_files = sorted(glob(os.path.join(DATA_DIR, 'train/pitch/*.npy')))
val_audio_files = sorted(glob(os.path.join(DATA_DIR, 'val/audio/*.npy')))
val_pitch_files = sorted(glob(os.path.join(DATA_DIR, 'val/pitch/*.npy')))

if not train_audio_files: print(f"FATAL: No training files found in '{DATA_DIR}'."); sys.exit(1)

# --- Hyperparameters ---
batch_size = 8
Nfft = 2048
win_size = 100
base_model_epochs = 100 
confidence_model_epochs = 50
learning_rate_base = 1e-4
learning_rate_confidence = 1e-4
gradient_clip_norm = 1.0

# --- Binning Hyperparameters ---
freq_min = 51.91; freq_max = 830.61; B = 96
num_semitones = int(B * np.log2(freq_max / freq_min))
bin_borders = [freq_min * np.power(2, i / B) for i in range(num_semitones + 1)]
bin_centers_log = np.array([(np.log2(bin_borders[i] / freq_min) + np.log2(bin_borders[i+1] / freq_min))/2 for i in range(len(bin_borders)-1)], dtype=np.float32)
num_bins = len(bin_borders) - 1
UNVOICED_CLASS_LABEL = num_bins

# --- Data Preprocessing ---
def load_and_preprocess_classification(audio_path, pitch_path):
    audio = np.load(audio_path.numpy().decode()).astype(np.float32)
    mean, std = np.mean(audio), np.std(audio)
    audio_norm = (audio - mean) / (std + 1e-7)
    pitch_hz = np.load(pitch_path.numpy().decode()).astype(np.float32)
    bin_indices = np.searchsorted(bin_borders, pitch_hz, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    class_labels = np.where(pitch_hz > 0, bin_indices, UNVOICED_CLASS_LABEL).astype(np.int32)
    return audio_norm, pitch_hz, class_labels

def prepare_dataset_classification(audio_files, pitch_files, b_size, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((audio_files, pitch_files))
    if shuffle: dataset = dataset.shuffle(buffer_size=len(audio_files))
    def _map_fn(audio_path, pitch_path):
        audio_norm, pitch_hz, class_labels = tf.py_function(load_and_preprocess_classification, [audio_path, pitch_path], [tf.float32, tf.float32, tf.int32])
        audio_norm.set_shape([win_size, 1025]); pitch_hz.set_shape([win_size]); class_labels.set_shape([win_size])
        return audio_norm, pitch_hz, class_labels
    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(b_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# --- Model Architectures ---
class ResNet_block(Model):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)); self.bn1 = BatchNormalization(); self.act1 = LeakyReLU(0.01)
        self.conv2 = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)); self.bn2 = BatchNormalization(); self.act2 = LeakyReLU(0.01)
        self.conv3 = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)); self.bn3 = BatchNormalization(); self.act3 = LeakyReLU(0.01)
        self.conv4 = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)); self.bn4 = BatchNormalization(); self.act4 = LeakyReLU(0.01)
        self.add = tf.keras.layers.Add(); self.pool = MaxPooling2D((1, 4))
    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor); shortcut = self.bn1(x, training=training); x = self.act1(shortcut); x = self.conv2(x); x = self.bn2(x, training=training); x = self.act2(x); x = self.conv3(x); x = self.bn3(x, training=training); x = self.act3(x); x = self.conv4(x); x = self.bn4(x, training=training); x = self.add([x, shortcut]); x = self.act4(x)
        return self.pool(x)

class BaselineClassificationModel(Model):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.rb1 = ResNet_block(32); self.rb2 = ResNet_block(64); self.rb3 = ResNet_block(128); self.rb4 = ResNet_block(256)
        self.dropout1 = Dropout(dropout_rate); self.reshape_layer = None
        self.dense = Dense(64, activation='relu', name='dense_head')
        self.output_classifier = Dense(num_bins + 1, activation='softmax', name='classifier_output')
    def call(self, x, training=False):
        x = self.rb1(x, training=training); x = self.rb2(x, training=training); x = self.dropout1(x, training=training); x = self.rb3(x, training=training); x = self.rb4(x, training=training)
        if self.reshape_layer is None: P = x.shape[2] * x.shape[3]; self.reshape_layer = Reshape((win_size, P))
        shared_features = self.reshape_layer(x); features = self.dense(shared_features)
        pitch_probs = self.output_classifier(features)
        return features, pitch_probs

class ConfidenceModel(Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.confidence_dense = Dense(256, activation='relu', name='confidence_dense')
        self.confidence_output = Dense(1, activation='sigmoid', name='confidence_output')
    def call(self, x, training=False):
        features, _ = self.base_model(x, training=False)
        features = tf.stop_gradient(features)
        c = self.confidence_dense(features)
        confidence_scores = self.confidence_output(c)
        return tf.squeeze(confidence_scores, axis=-1)

# --- Helper Functions ---
def compute_metrics(y_true_hz, y_pred_hz):
    rpa, rca, oa = [], [], [];
    for i in range(y_true_hz.shape[0]):
        gfv, efv = y_true_hz[i], y_pred_hz[i]; t = np.arange(len(gfv)) * 0.01
        try:
            ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(t, gfv, t, efv)
            rpa.append(mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)); rca.append(mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)); oa.append(mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c))
        except: continue
    return np.mean(rpa) if rpa else 0, np.mean(rca) if rca else 0, np.mean(oa) if oa else 0

def calculate_hz_from_classification(probs):
    pred_indices = tf.argmax(probs, axis=-1, output_type=tf.int32)
    log_freq_voiced = tf.gather(bin_centers_log, pred_indices)
    hz_values = freq_min * tf.pow(2.0, log_freq_voiced)
    is_voiced_mask = tf.cast(pred_indices != UNVOICED_CLASS_LABEL, dtype=tf.float32)
    return hz_values * is_voiced_mask
    
def calculate_tcp_n_target(probs, true_labels, epsilon=1e-9):
    prob_of_predicted_class = tf.reduce_max(probs, axis=-1)
    true_labels_one_hot = tf.one_hot(true_labels, depth=tf.shape(probs)[-1])
    prob_of_true_class = tf.reduce_sum(probs * true_labels_one_hot, axis=-1)
    tcp_n = prob_of_true_class / (prob_of_predicted_class + epsilon)
    is_correct_mask = tf.cast(tf.argmax(probs, axis=-1, output_type=tf.int32) == true_labels, dtype=tf.float32)
    return tcp_n * (1.0 - is_correct_mask) + is_correct_mask

#############################################################################
# --- PART 1: TRAIN THE BASE CLASSIFICATION MODEL ---
#############################################################################
print("--- PART 1: Starting Base Classification Model Training ---")

base_model = BaselineClassificationModel()
optimizer_base = keras.optimizers.Adam(learning_rate=learning_rate_base)
loss_fn_base = tf.keras.losses.SparseCategoricalCrossentropy()
dummy_input = tf.zeros((1, win_size, Nfft // 2 + 1, 1)); _ = base_model(dummy_input)
print("Base classification model built successfully.")

@tf.function
def train_step_base(x_audio, y_labels):
    with tf.GradientTape() as tape:
        _, probs = base_model(x_audio, training=True)
        loss = loss_fn_base(y_labels, probs)
    grads = tape.gradient(loss, base_model.trainable_variables)
    grads = [tf.clip_by_norm(g, gradient_clip_norm) if g is not None else None for g in grads]
    optimizer_base.apply_gradients(zip(grads, base_model.trainable_variables))
    y_pred_hz = calculate_hz_from_classification(probs)
    return loss, y_pred_hz

@tf.function
def test_step_base(x_audio, y_labels):
    _, probs = base_model(x_audio, training=False)
    loss = loss_fn_base(y_labels, probs)
    y_pred_hz = calculate_hz_from_classification(probs)
    return loss, y_pred_hz

train_dataset = prepare_dataset_classification(train_audio_files, train_pitch_files, batch_size, shuffle=True)
val_dataset = prepare_dataset_classification(val_audio_files, val_pitch_files, batch_size, shuffle=False)

for epoch in range(base_model_epochs):
    print(f'\nEpoch {epoch + 1}/{base_model_epochs}')
    train_loss_metric = tf.keras.metrics.Mean(); train_oa = []
    for x_batch, y_hz_batch, y_labels_batch in tqdm(train_dataset, desc="Training Base Model"):
        x_batch = x_batch[..., tf.newaxis]; loss, y_pred_hz = train_step_base(x_batch, y_labels_batch)
        train_loss_metric.update_state(loss); _, _, o = compute_metrics(y_hz_batch.numpy(), y_pred_hz.numpy()); train_oa.append(o)
    print(f"  Base Model Train Loss: {train_loss_metric.result():.4f} | OA: {np.mean(train_oa):.4f}")

    val_loss_metric = tf.keras.metrics.Mean(); val_oa = []
    for x_batch, y_hz_batch, y_labels_batch in tqdm(val_dataset, desc="Validating Base Model"):
        x_batch = x_batch[..., tf.newaxis]; loss, y_pred_hz = test_step_base(x_batch, y_labels_batch)
        val_loss_metric.update_state(loss); _, _, o = compute_metrics(y_hz_batch.numpy(), y_pred_hz.numpy()); val_oa.append(o)
    print(f"  Base Model Validation Loss: {val_loss_metric.result():.4f} | OA: {np.mean(val_oa):.4f}")

    if (epoch + 1) % 10 == 0 or (epoch + 1) == base_model_epochs:
        save_path = os.path.join(BASE_MODEL_WEIGHTS_PATH, f'baseline_model_{epoch + 1}.weights.h5')
        base_model.save_weights(save_path)
        print(f"  Base model weights saved to {save_path}")

print("\n--- Base Model Training Complete ---")


#############################################################################
# --- PART 2: TRAIN THE CONFIDENCE MODEL ---
#############################################################################
print("\n\n--- PART 2: Starting Confidence Model Training ---")

# Re-instantiate the base model to ensure a clean state, then load weights
base_model_frozen = BaselineClassificationModel()
_, _ = base_model_frozen(dummy_input)
final_base_weights_path = os.path.join(BASE_MODEL_WEIGHTS_PATH, f'baseline_model_{base_model_epochs}.weights.h5')
base_model_frozen.load_weights(final_base_weights_path)
print(f"Loaded final base model weights from {final_base_weights_path}")
base_model_frozen.trainable = False
print("Base model has been frozen.")

confidence_model = ConfidenceModel(base_model_frozen)
_ = confidence_model(dummy_input)
print("Confidence model built successfully.")

optimizer_confidence = keras.optimizers.Adam(learning_rate=learning_rate_confidence)
loss_fn_confidence = tf.keras.losses.MeanSquaredError()

@tf.function
def train_step_confidence(x_audio, y_true_labels):
    _, probs = base_model_frozen(x_audio, training=False)
    y_true_confidence = calculate_tcp_n_target(probs, y_true_labels)
    with tf.GradientTape() as tape:
        y_pred_confidence = confidence_model(x_audio, training=True)
        loss = loss_fn_confidence(y_true_confidence, y_pred_confidence)
    grads = tape.gradient(loss, confidence_model.trainable_variables)
    optimizer_confidence.apply_gradients(zip(grads, confidence_model.trainable_variables))
    return loss

for epoch in range(confidence_model_epochs):
    print(f'\nEpoch {epoch + 1}/{confidence_model_epochs}')
    train_loss_metric = tf.keras.metrics.Mean()
    for x_batch, _, y_labels_batch in tqdm(train_dataset, desc="Training Confidence Model"):
        x_batch = x_batch[..., tf.newaxis]
        loss = train_step_confidence(x_batch, y_labels_batch)
        train_loss_metric.update_state(loss)
    print(f"  Confidence Model Train Loss (MSE): {train_loss_metric.result():.4f}")
    if (epoch + 1) % 10 == 0 or (epoch + 1) == confidence_model_epochs:
        save_path = os.path.join(CONFIDENCE_MODEL_WEIGHTS_PATH, f'confidence_model_{epoch + 1}.weights.h5')
        confidence_model.save_weights(save_path)
        print(f"  Confidence model weights saved to {save_path}")

print("\n--- ALL TRAINING COMPLETE ---")
print(f"Base model weights are in: {BASE_MODEL_WEIGHTS_PATH}")
print(f"Confidence model weights are in: {CONFIDENCE_MODEL_WEIGHTS_PATH}")
print("You can now proceed to the fine-tuning script.")