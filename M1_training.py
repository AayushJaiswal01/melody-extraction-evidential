

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Reshape, LeakyReLU
from tensorflow.keras import Model
import random
from glob import glob
from tqdm import tqdm
import mir_eval.melody
import keras
import sys

# --- Configuration & File Paths ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try: tf.config.experimental.set_memory_growth(physical_devices[0], True); print("GPU memory growth set")
    except Exception as e: print(e)

# --- CHOOSE YOUR BASELINE EXPERIMENT ---
BETA_VALUE = 0.5  # Options: 0.0 (standard NLL), 0.5 (recommended), 1.0 (MSE-like)

# --- PATHS ---
DATA_DIR = '../../../../../spl_v2_mir1k/npy_data'
WEIGHTS_PATH = f'./saved_models_beta_nll_baseline/beta_{BETA_VALUE}/'
os.makedirs(WEIGHTS_PATH, exist_ok=True)

train_audio_files = sorted(glob(os.path.join(DATA_DIR, 'train/audio/*.npy')))
train_pitch_files = sorted(glob(os.path.join(DATA_DIR, 'train/pitch/*.npy')))
val_audio_files = sorted(glob(os.path.join(DATA_DIR, 'val/audio/*.npy')))
val_pitch_files = sorted(glob(os.path.join(DATA_DIR, 'val/pitch/*.npy')))

if not train_audio_files: print(f"!!! WARNING: No training files found. Check path."); sys.exit(1)

# --- Hyperparameters ---
batch_size = 16; Nfft = 2048; win_size = 100; epochs = 200; learning_rate = 1e-4

# --- Binning ---
freq_min = 51.91; freq_max = 830.61; B = 96
num_semitones = int(B * np.log2(freq_max / freq_min))
bin_borders = [freq_min * np.power(2, i / B) for i in range(num_semitones + 1)]
bin_centers_log = np.array([(np.log2(bin_borders[i] / freq_min) + np.log2(bin_borders[i+1] / freq_min))/2 for i in range(len(bin_borders)-1)], dtype=np.float32)
num_bins = len(bin_centers_log)

# --- Global Stats & Data Preprocessing ---
def calculate_global_mean_std_runtime(audio_file_paths):
    print(f"Calculating global mean and std from {len(audio_file_paths)} audio files...")
    count, mean_acc, m2_acc = 0, 0.0, 0.0
    for f_path in tqdm(audio_file_paths, desc="Calculating global stats"):
        try:
            audio = np.load(f_path).astype(np.float64); n = audio.size
            if n == 0: continue
            delta = audio - mean_acc; mean_acc += np.sum(delta) / (count + n); delta2 = audio - mean_acc; m2_acc += np.sum(delta * delta2); count += n
        except Exception: continue
    if count < 2: return np.float32(0.0), np.float32(1.0)
    global_mean = mean_acc; global_std = np.sqrt(m2_acc / (count - 1))
    print(f"Global mean: {global_mean:.4f}, Global std: {global_std:.4f}")
    return np.float32(global_mean), np.float32(global_std)
global_mean_np, global_std_np = calculate_global_mean_std_runtime(train_audio_files)

def load_wav_global_norm(wav_path):
    audio = np.load(wav_path.numpy().decode()).astype(np.float32)
    return (audio - global_mean_np) / (global_std_np + 1e-6)
def load_pitch_and_bins(pitch_path):
    pitch_hz = np.load(pitch_path.numpy().decode()).astype(np.float32)
    voicing = (pitch_hz > 0).astype(np.float32)
    bin_indices_np = np.searchsorted(bin_borders, pitch_hz, side='right') - 1
    bin_indices_np = np.clip(bin_indices_np, 0, num_bins - 1)
    bin_indices_np = bin_indices_np * voicing
    return pitch_hz, voicing, bin_indices_np.astype(np.float32)
def prepare_dataset_binned(audio_files, pitch_files, b_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((audio_files, pitch_files))
    if shuffle: dataset = dataset.shuffle(buffer_size=len(audio_files))
    def _map_fn(wav_path, pitch_path):
        audio = tf.py_function(load_wav_global_norm, [wav_path], tf.float32)
        pitch_hz, voicing, bin_indices = tf.py_function(load_pitch_and_bins, [pitch_path], [tf.float32, tf.float32, tf.float32])
        audio.set_shape([win_size, Nfft//2+1]); pitch_hz.set_shape([win_size]); voicing.set_shape([win_size]); bin_indices.set_shape([win_size])
        return audio, pitch_hz, voicing, bin_indices
    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(b_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# --- Metric calculation ---
def compute_metrics(y_true_hz_batch, y_pred_hz_batch):
    rpa_list, rca_list, oa_list = [], [], []
    for i in range(y_true_hz_batch.shape[0]):
        gfv = y_true_hz_batch[i]; efv = y_pred_hz_batch[i]; t = np.arange(len(gfv)) * 0.01
        try: ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(t, gfv, t, efv); rpa_list.append(mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)); rca_list.append(mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)); oa_list.append(mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c))
        except: continue
    return np.mean(rpa_list) if rpa_list else 0, np.mean(rca_list) if rca_list else 0, np.mean(oa_list) if oa_list else 0

# --- β-NLL Loss Function (from the paper) ---
def beta_nll_loss(y_true_bins, y_true_voicing, pitch_params, beta):
    mu_pred, sigma_pred = pitch_params
    y_true_bins_f32 = tf.cast(y_true_bins, tf.float32)
    y_true_voicing_f32 = tf.cast(y_true_voicing, tf.float32)
    
    # Calculate standard NLL for each frame
    variance = tf.square(sigma_pred)
    nll_per_frame = 0.5 * (tf.math.log(2.0 * np.pi * variance + 1e-9) + tf.square(y_true_bins_f32 - mu_pred) / (variance + 1e-9))
    
    # Apply the beta weighting from the paper
    # Use tf.stop_gradient on the weighting term
    weighting_term = tf.stop_gradient(tf.pow(variance, beta))
    weighted_nll_per_frame = weighting_term * nll_per_frame
    
    # Mask the loss to only apply to voiced frames
    voiced_mask = y_true_voicing_f32
    masked_nll = weighted_nll_per_frame * voiced_mask
    
    # Normalize by the number of voiced frames
    num_voiced_frames = tf.reduce_sum(voiced_mask) + 1e-9
    loss = tf.reduce_sum(masked_nll) / num_voiced_frames
    return loss

# --- NLL Model Architecture (Single Head) ---
class ResNet_block(Model):
    def __init__(self, filters):
        super().__init__(); self.conv1=Conv2D(filters,(3,3),padding='same',kernel_initializer='he_normal'); self.bn1=BatchNormalization(); self.act1=LeakyReLU(0.01); self.conv2=Conv2D(filters,(3,3),padding='same',kernel_initializer='he_normal'); self.bn2=BatchNormalization(); self.act2=LeakyReLU(0.01); self.add=tf.keras.layers.Add(); self.pool=MaxPooling2D((1,4))
    def call(self, input_tensor, training=False):
        x=self.conv1(input_tensor); shortcut=self.bn1(x,training=training); x=self.act1(shortcut)
        x=self.conv2(x); x=self.bn2(x,training=training); x=self.add([x,shortcut]); x=self.act2(x); return self.pool(x)

class NLLMelodyModel(Model):
    def __init__(self):
        super().__init__()
        self.rb1 = ResNet_block(32); self.rb2 = ResNet_block(64); self.rb3 = ResNet_block(128); self.rb4 = ResNet_block(256)
        self.reshape_layer = None
        self.pitch_dense = Dense(64, activation='relu', name='pitch_dense')
        self.pitch_output = Dense(2, name='pitch_output')
    def call(self, x, training=False):
        x = self.rb1(x,training=training); x = self.rb2(x,training=training); x = self.rb3(x,training=training); x = self.rb4(x,training=training)
        if self.reshape_layer is None: P = x.shape[2]*x.shape[3]; self.reshape_layer = Reshape((win_size,P))
        shared_features = self.reshape_layer(x); p = self.pitch_dense(shared_features); pitch_params_raw = self.pitch_output(p)
        mu = tf.math.sigmoid(pitch_params_raw[:,:,0]) * float(num_bins)
        sigma = tf.nn.softplus(pitch_params_raw[:,:,1]) + 1e-6
        return mu, sigma

# --- Training Setup ---
model = NLLMelodyModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
def calculate_expected_value_binned(mu_pred):
    voicing_decision = tf.cast(mu_pred > 0.5, tf.float32)
    pred_indices = tf.cast(tf.round(tf.clip_by_value(mu_pred, 0, num_bins - 1)), dtype=tf.int32)
    pred_log_freq = tf.gather(bin_centers_log, pred_indices)
    final_pitch_hz = freq_min * tf.pow(2.0, pred_log_freq)
    return final_pitch_hz * voicing_decision
@tf.function
def train_step(x_audio, y_bins, y_voicing):
    with tf.GradientTape() as tape:
        mu_pred, sigma_pred = model(x_audio, training=True)
        loss = beta_nll_loss(y_bins, y_voicing, (mu_pred, sigma_pred), beta=BETA_VALUE)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    y_pred_hz = calculate_expected_value_binned(mu_pred)
    return loss, y_pred_hz
@tf.function
def test_step(x_audio, y_bins, y_voicing):
    mu_pred, sigma_pred = model(x_audio, training=False)
    loss = beta_nll_loss(y_bins, y_voicing, (mu_pred, sigma_pred), beta=BETA_VALUE)
    y_pred_hz = calculate_expected_value_binned(mu_pred)
    return loss, y_pred_hz

# --- Main Training Loop ---
train_dataset = prepare_dataset_binned(train_audio_files, train_pitch_files, batch_size, shuffle=True)
val_dataset = prepare_dataset_binned(val_audio_files, val_pitch_files, batch_size, shuffle=False)
for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs} (beta = {BETA_VALUE})')
    train_loss = tf.keras.metrics.Mean(); train_rpa, train_rca, train_oa = [],[],[]
    for x_batch, y_hz_batch, y_voicing_batch, y_bins_batch in tqdm(train_dataset, desc="Training"):
        x_batch = x_batch[..., tf.newaxis]
        loss, y_pred_hz = train_step(x_batch, y_bins_batch, y_voicing_batch)
        train_loss.update_state(loss)
        r, c, o = compute_metrics(y_hz_batch.numpy(), y_pred_hz.numpy())
        train_rpa.append(r); train_rca.append(c); train_oa.append(o)
    print(f"Train Loss: {train_loss.result():.4f} | OA: {np.mean(train_oa):.4f}")
    val_loss = tf.keras.metrics.Mean(); val_rpa, val_rca, val_oa = [],[],[]
    for x_batch, y_hz_batch, y_voicing_batch, y_bins_batch in tqdm(val_dataset, desc="Validating"):
        x_batch = x_batch[..., tf.newaxis]
        loss, y_pred_hz = test_step(x_batch, y_bins_batch, y_voicing_batch)
        val_loss.update_state(loss)
        r, c, o = compute_metrics(y_hz_batch.numpy(), y_pred_hz.numpy())
        val_rpa.append(r); val_rca.append(c); val_oa.append(o)
    print(f"Validation Loss: {val_loss.result():.4f} | OA: {np.mean(val_oa):.4f}")
    if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
        save_path = os.path.join(WEIGHTS_PATH, f'beta_nll_model_{epoch + 1}.weights.h5')
        model.save_weights(save_path)
        print(f"Model weights saved to {save_path}")