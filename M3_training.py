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
# Define a new directory for the M3 model weights
WEIGHTS_PATH = './model_weights/TRUE_SINGLE_HEAD_EDL_MODEL/' 
os.makedirs(WEIGHTS_PATH, exist_ok=True)

# --- Data Loading ---
train_audio_files = sorted(glob(os.path.join(DATA_DIR, 'train/audio/*.npy')))
train_pitch_files = sorted(glob(os.path.join(DATA_DIR, 'train/pitch/*.npy')))
val_audio_files = sorted(glob(os.path.join(DATA_DIR, 'val/audio/*.npy')))
val_pitch_files = sorted(glob(os.path.join(DATA_DIR, 'val/pitch/*.npy')))

if not train_audio_files: print(f"FATAL: No training files found in '{DATA_DIR}'."); sys.exit(1)
if not val_audio_files: print(f"FATAL: No validation files found in '{DATA_DIR}'."); sys.exit(1)

# --- Hyperparameters ---
batch_size = 8
Nfft = 2048
win_size = 100
epochs = 300
learning_rate = 1e-4
gradient_clip_norm = 1.0

# --- Binning Hyperparameters ---
freq_min = 51.91; freq_max = 830.61; B = 96
num_semitones = int(B * np.log2(freq_max / freq_min))
bin_borders = [freq_min * np.power(2, i / B) for i in range(num_semitones + 1)]
bin_centers_log = np.array([(np.log2(bin_borders[i] / freq_min) + np.log2(bin_borders[i+1] / freq_min))/2 for i in range(len(bin_borders)-1)], dtype=np.float32)
num_bins = len(bin_centers_log)

# --- Data Preprocessing ---
bin_borders_tf = tf.constant(bin_borders, dtype=tf.float32)
def load_and_normalize_per_file(wav_path):
    audio = np.load(wav_path.numpy().decode()).astype(np.float32)
    mean, std = np.mean(audio), np.std(audio)
    return (audio - mean) / (std + 1e-7)

def load_pitch_numpy(pitch_path):
    return np.load(pitch_path.numpy().decode()).astype(np.float32)

def load_and_preprocess(audio_path, pitch_path):
    audio_norm = tf.py_function(load_and_normalize_per_file, [audio_path], tf.float32)
    pitch_hz = tf.py_function(load_pitch_numpy, [pitch_path], tf.float32)
    audio_norm.set_shape([win_size, 1025])
    pitch_hz.set_shape([win_size])
    bin_indices = tf.searchsorted(bin_borders_tf, pitch_hz, side='right') - 1
    bin_indices = tf.clip_by_value(bin_indices, 0, num_bins - 1)
    is_voiced = tf.cast(pitch_hz > 0, dtype=tf.int32)
    # Target is bin_index for voiced frames and 0 for unvoiced frames.
    bin_indices_target = bin_indices * is_voiced 
    return audio_norm, pitch_hz, tf.cast(bin_indices_target, tf.float32)

def prepare_dataset(audio_files, pitch_files, b_size, shuffle=False):
    if not audio_files: return None
    dataset = tf.data.Dataset.from_tensor_slices((audio_files, pitch_files))
    if shuffle: dataset = dataset.shuffle(buffer_size=len(audio_files))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(b_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# --- Model Architecture (M3 Version) ---
class ResNet_block(Model):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))
        self.bn1 = BatchNormalization(); self.act1 = LeakyReLU(0.01)
        self.conv2 = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))
        self.bn2 = BatchNormalization(); self.act2 = LeakyReLU(0.01)
        self.conv3 = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))
        self.bn3 = BatchNormalization(); self.act3 = LeakyReLU(0.01)
        self.conv4 = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5))
        self.bn4 = BatchNormalization(); self.act4 = LeakyReLU(0.01)
        self.add = tf.keras.layers.Add(); self.pool = MaxPooling2D((1, 4))
    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor); shortcut = self.bn1(x, training=training); x = self.act1(shortcut)
        x = self.conv2(x); x = self.bn2(x, training=training); x = self.act2(x)
        x = self.conv3(x); x = self.bn3(x, training=training); x = self.act3(x)
        x = self.conv4(x); x = self.bn4(x, training=training); x = self.add([x, shortcut]); x = self.act4(x)
        return self.pool(x)

class MelodyModel(Model):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.rb1 = ResNet_block(32); self.rb2 = ResNet_block(64); self.rb3 = ResNet_block(128); self.rb4 = ResNet_block(256)
        self.dropout1 = Dropout(dropout_rate); self.reshape_layer = None
        self.pitch_dense = Dense(64, activation='relu', name='pitch_dense')
        self.pitch_output = TimeDistributed(Dense(4, activation=None), name='pitch_nig_params')
    def call(self, x, training=False):
        x = self.rb1(x, training=training); x = self.rb2(x, training=training); x = self.dropout1(x, training=training)
        x = self.rb3(x, training=training); x = self.rb4(x, training=training)
        if self.reshape_layer is None: P = x.shape[2] * x.shape[3]; self.reshape_layer = Reshape((win_size, P))
        shared_features = self.reshape_layer(x)
        pitch_features = self.pitch_dense(shared_features); nig_params_raw = self.pitch_output(pitch_features)
        
        gamma = nig_params_raw[..., 0] 
        nu = tf.nn.softplus(nig_params_raw[..., 1]) + 1e-6
        alpha = tf.nn.softplus(nig_params_raw[..., 2]) + 1.1
        beta = tf.nn.softplus(nig_params_raw[..., 3]) + 1e-6
        return (gamma, nu, alpha, beta)

# --- Loss and Metrics ---
def evidential_loss_regression(y_true, nig_params):
    gamma, nu, alpha, beta = nig_params; error = tf.abs(y_true - gamma); two_beta_nu = 2 * beta * (1 + nu)
    nll = 0.5 * tf.math.log(np.pi / nu) - alpha * tf.math.log(two_beta_nu) + (alpha + 0.5) * tf.math.log(nu * tf.square(error) + two_beta_nu) + tf.math.lgamma(alpha) - tf.math.lgamma(alpha + 0.5)
    reg = error * (2 * nu + alpha); return nll, reg

def total_evidential_loss_m3(y_bin_indices, nig_params, reg_w=0.01):
    nll_loss, reg_loss = evidential_loss_regression(y_bin_indices, nig_params)
    # The loss is the mean of the NLL and regularizer across ALL frames in the batch
    mean_nll = tf.reduce_mean(nll_loss)
    mean_reg = tf.reduce_mean(reg_loss)
    return mean_nll + reg_w * mean_reg

def compute_metrics(y_true_hz, y_pred_hz):
    rpa, rca, oa = [], [], []
    for i in range(y_true_hz.shape[0]):
        gfv, efv = y_true_hz[i], y_pred_hz[i]
        t = np.arange(len(gfv)) * 0.01
        try:
            ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(t, gfv, t, efv)
            rpa.append(mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c))
            rca.append(mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c))
            oa.append(mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c))
        except:
            continue
    return np.mean(rpa) if rpa else 0, np.mean(rca) if rca else 0, np.mean(oa) if oa else 0

def calculate_expected_value(nig_params):
    gamma, _, _, _ = nig_params
    pred_indices = tf.cast(tf.round(tf.clip_by_value(gamma, 0, num_bins - 1)), dtype=tf.int32)
    pred_log_freq = tf.gather(bin_centers_log, pred_indices)
    exp_val = freq_min * tf.pow(2.0, pred_log_freq)
    # Infer voicing from the pitch prediction itself (if gamma > 0.5, it's likely a voiced bin)
    mask = tf.cast(gamma > 0.5, tf.float32)
    return exp_val * mask

# --- Training Setup ---
model = MelodyModel()
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Build model by passing a dummy input
dummy_input = tf.zeros((1, win_size, Nfft // 2 + 1, 1))
_ = model(dummy_input, training=False)
print("M3 Model built VERY successfully.")

@tf.function
def train_step(x_audio, y_bins):
    with tf.GradientTape() as tape:
        nig_params = model(x_audio, training=True)
        loss = total_evidential_loss_m3(y_bins, nig_params)
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, gradient_clip_norm) if g is not None else None for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    y_pred_hz = calculate_expected_value(nig_params)
    return loss, y_pred_hz

@tf.function
def test_step(x_audio, y_bins):
    nig_params = model(x_audio, training=False)
    loss = total_evidential_loss_m3(y_bins, nig_params)
    y_pred_hz = calculate_expected_value(nig_params)
    return loss, y_pred_hz

# --- Main Training Loop ---
train_dataset = prepare_dataset(train_audio_files, train_pitch_files, batch_size, shuffle=True)
val_dataset = prepare_dataset(val_audio_files, val_pitch_files, batch_size, shuffle=False)

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')
    
    # --- Training Phase ---
    train_loss_metric = tf.keras.metrics.Mean()
    train_rpa, train_rca, train_oa = [], [], []
    for x_batch, y_hz_batch, y_bins_batch in tqdm(train_dataset, desc="Training"):
        x_batch = x_batch[..., tf.newaxis]
        loss, y_pred_hz = train_step(x_batch, y_bins_batch)
        train_loss_metric.update_state(loss)
        r, c, o = compute_metrics(y_hz_batch.numpy(), y_pred_hz.numpy())
        train_rpa.append(r); train_rca.append(c); train_oa.append(o)
    
    print(f"  Train Loss: {train_loss_metric.result():.4f} | OA: {np.mean(train_oa):.4f} | RPA: {np.mean(train_rpa):.4f} | RCA: {np.mean(train_rca):.4f}")

    # --- Validation Phase ---
    val_loss_metric = tf.keras.metrics.Mean()
    val_rpa, val_rca, val_oa = [], [], []
    for x_batch, y_hz_batch, y_bins_batch in tqdm(val_dataset, desc="Validating"):
        x_batch = x_batch[..., tf.newaxis]
        loss, y_pred_hz = test_step(x_batch, y_bins_batch)
        val_loss_metric.update_state(loss)
        r, c, o = compute_metrics(y_hz_batch.numpy(), y_pred_hz.numpy())
        val_rpa.append(r); val_rca.append(c); val_oa.append(o)

    print(f"  Validation Loss: {val_loss_metric.result():.4f} | OA: {np.mean(val_oa):.4f} | RPA: {np.mean(val_rpa):.4f} | RCA: {np.mean(val_rca):.4f}")

    # Save model weights periodically
    if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
        save_path = os.path.join(WEIGHTS_PATH, f'single_head_edl_model_{epoch + 1}.weights.h5')
        model.save_weights(save_path)
        print(f"  Model weights saved to {save_path}")

print("\n--- Training Complete ---")