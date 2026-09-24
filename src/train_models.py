# -*- coding: utf-8 -*-
"""
Low-contrast microscopy surface classification — automated multi-seed training script.
One-to-one .py export of plastik-modeller_SEED-FINAL-v3-auto.ipynb (cleaned).

Run (from the 'Kaynak Kodlar' folder; dataset at ../plastik_mikroskop_veriseti/):
    nohup python -u plastik_modeller_SEED_FINAL_v3_auto.py > egitim_log.txt 2>&1 &

Seeds are configured at the top (SEEDS = [42, 7, 123]).
Set SEEDS = [42] first if you want the validation run only.
"""


# ======================================================================
# CELL 0
# ======================================================================
# ============================ Imports and configuration ============================
import os
import random
import numpy as np
import tensorflow as tf

# Random seeds for the multi-seed study (the driver runs all of them in order).
SEEDS = [42, 7, 123]           # 42 first: cross-check against the originally reported results
SEED_VALUE = SEEDS[0]          # updated by set_seed() each iteration

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def set_seed(seed):
    """Fix all RNG seeds for the given value and update the global SEED_VALUE."""
    global SEED_VALUE
    SEED_VALUE = seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seeds fixed: {seed}")

set_seed(SEED_VALUE)
# -------------------------------------------------------

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models, mixed_precision, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Per-architecture ImageNet preprocessing functions
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionv3_preprocess
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess

# ------------------------------------------------------------------------------
# 1. Optimizer selection
# ------------------------------------------------------------------------------
try:
    from tensorflow.keras.optimizers.legacy import AdamW
    print("Legacy (M1/M2-optimized) AdamW loaded.")
except ImportError:
    from tensorflow.keras.optimizers import AdamW
    print("Legacy optimizer not available; using standard AdamW.")

# ------------------------------------------------------------------------------
# 2. GPU configuration
# ------------------------------------------------------------------------------
print(f"TensorFlow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected: {gpus}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found; running on CPU.")

# ------------------------------------------------------------------------------
# 3. Paths and constants
# ------------------------------------------------------------------------------
dataset_base_path = '../plastik_mikroskop_veriseti/'


train_dataset_path = os.path.join(dataset_base_path, 'train')
validation_dataset_path = os.path.join(dataset_base_path, 'val')
predict_dataset_path = os.path.join(dataset_base_path, 'test')

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

BATCH_SIZE = 32 
EPOCHS = 200

# output_dir is set per seed inside the driver (run_seed).
output_dir = f"sonuclar_nisan2026_best_models_seed{SEED_VALUE}"
os.makedirs(output_dir, exist_ok=True)

# Head L2 regularization flag (set automatically per stage).
# Stage 1 (feature extraction): no head L2  -> USE_HEAD_L2 = False
# Stage 2 (fine-tuning): head L2 = 0.001    -> USE_HEAD_L2 = True
# Set automatically by the driver for each stage; no manual change needed.
USE_HEAD_L2 = False


# ======================================================================
# CELL 1
# ======================================================================
# ============================ Data loading and normalization ============================

def get_data_augmentation():
    # Augmentation layers are seeded for reproducibility
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=SEED_VALUE), 
            layers.RandomRotation(0.2, seed=SEED_VALUE),
            layers.RandomZoom(0.2, seed=SEED_VALUE),
            layers.RandomContrast(0.1, seed=SEED_VALUE),   
            layers.RandomBrightness(0.1, seed=SEED_VALUE), 
        ],
        name="data_augmentation",
    )
    return data_augmentation

def to_rgb(x, y):
    """
    Return 3-channel RGB with a batch-aware static shape.
    """
    x = tf.cast(x, tf.float32) # cast to float (no scaling)
    
    channels = tf.shape(x)[-1]
    x = tf.cond(tf.equal(channels, 1), lambda: tf.repeat(x, repeats=3, axis=-1), lambda: x)
    x = tf.cond(tf.greater(channels, 3), lambda: x[..., :3], lambda: x)
    
    # keep the batch dimension dynamic (None)
    x.set_shape([None, IMG_HEIGHT, IMG_WIDTH, 3])
    
    return x, y

def load_datasets():
    print("Loading datasets (fixed seed)...")
    
    # Train/val shuffles are seeded
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dataset_path, 
        image_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        label_mode='int', 
        shuffle=True, 
        seed=SEED_VALUE  # seeded shuffle
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dataset_path, 
        image_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        label_mode='int', 
        shuffle=True, 
        seed=SEED_VALUE  # seeded shuffle
    )
    
    # Test set is not shuffled
    test_ds = tf.keras.utils.image_dataset_from_directory(
        predict_dataset_path, 
        image_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        label_mode='int', 
        shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Classes: {class_names} (total {num_classes})")

    # RGB conversion
    train_ds = train_ds.map(to_rgb, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(to_rgb, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(to_rgb, num_parallel_calls=tf.data.AUTOTUNE)

    # Performance (prefetch)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names, num_classes

# ======================================================================
# CELL 2
# ======================================================================
# ============================ Model definitions ============================
# Head regularization by stage: None (feature extraction) or L2(0.001) (fine-tuning).
def _head_l2():
    return regularizers.l2(0.001) if USE_HEAD_L2 else None
# LeNet-5 (from scratch)
def build_lenet5_scratch(num_classes, norm_layer):
    
    # Input shape (height, width, channels)
    INPUT_SHAPE = (224, 224, 3) 
    
    model = models.Sequential(name="LeNet-5-Scratch")
    
    model.add(layers.Input(shape=INPUT_SHAPE))
    
    # Data augmentation
    model.add(get_data_augmentation()) 
    
    # Z-score normalization
    model.add(norm_layer)
    
    # Resize to 32x32 (LeNet-5's native input size)
    model.add(layers.Resizing(32, 32))
    
    # LeNet-5 body (all layers trainable)
    model.add(layers.Conv2D(6, kernel_size=(5, 5), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model
# (Unused CIFAR-LeNet and AlexNet builders were removed.)

# VGG16 (ImageNet transfer, frozen base)
def build_vgg16(num_classes, norm_layer):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = get_data_augmentation()(inputs)
    x = layers.Lambda(vgg16_preprocess, name='preprocess_vgg16')(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=_head_l2())(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="VGG16")

# VGG19 (ImageNet transfer, frozen base)
def build_vgg19(num_classes, norm_layer):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False 
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = get_data_augmentation()(inputs)
    x = layers.Lambda(vgg19_preprocess, name='preprocess_vgg19')(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=_head_l2())(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="VGG19")

# ResNet50 (ImageNet transfer, frozen base)
def build_resnet50(num_classes, norm_layer):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = get_data_augmentation()(inputs)
    x = layers.Lambda(resnet50_preprocess, name='preprocess_resnet50')(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=_head_l2())(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="ResNet50")

# InceptionV3 (ImageNet transfer, frozen base)
def build_inceptionv3(num_classes, norm_layer):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = get_data_augmentation()(inputs)
    x = layers.Lambda(inceptionv3_preprocess, name='preprocess_inception')(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=_head_l2())(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="InceptionV3")

# EfficientNetB0 (ImageNet transfer, frozen base)
def build_efficientnetb0(num_classes, norm_layer):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = get_data_augmentation()(inputs)
    x = layers.Lambda(efficientnet_preprocess, name='preprocess_efficientnet')(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=_head_l2())(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="EfficientNetB0")

# ResNet50 (from scratch) - ablation: same architecture, no pretraining
def build_resnet50_scratch(num_classes, norm_layer):
    base_model = ResNet50(weights=None, include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = True                 # all layers trained from scratch
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = get_data_augmentation()(inputs)
    x = norm_layer(x)                           # from-scratch models use Z-score (not ImageNet preprocessing)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=_head_l2())(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="ResNet50_Scratch")

# CustomModernNet (from scratch)
def build_custom_modern_net_base(num_classes, norm_layer):
    from tensorflow.keras.layers import (Input, SeparableConv2D, BatchNormalization,
                                         LayerNormalization, Activation,
                                         GlobalAveragePooling2D, Dense, Dropout)
    from tensorflow.keras.models import Model
    
    inputs = Input(shape=INPUT_SHAPE, name="input")
    x = get_data_augmentation()(inputs)
    x = norm_layer(x) # Z-score normalization
    
    def block(x, filters, strides=1):
        x = SeparableConv2D(filters, 3, padding='same', strides=strides, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
        
    x = block(x, 48, 1)
    x = block(x, 96, 2)
    x = block(x, 160, 2)
    x = block(x, 224, 2)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=_head_l2())(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='Custom_Modern_Net')

# ======================================================================
# CELL 3
# ======================================================================
# ============================ Training and reporting ============================

def train_robust_model(model_func, model_name, num_classes, train_ds, val_ds, norm_layer):
    print(f"\n--- Training (feature extraction): {model_name} ---")
    checkpoint_path = os.path.join(output_dir, f"{model_name}_best_model.h5")
    log_path = os.path.join(output_dir, f"{model_name}_training_log.csv")

    model = model_func(num_classes, norm_layer)
    optimizer = AdamW(learning_rate=3e-4, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        CSVLogger(log_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
    ]
    
    initial_epoch = 0
    if os.path.exists(log_path) and os.path.exists(checkpoint_path):
        try:
            log_df = pd.read_csv(log_path)
            if not log_df.empty:
                initial_epoch = log_df['epoch'].iloc[-1] + 1
                print(f"Resuming training from epoch {initial_epoch}.")
                model.load_weights(checkpoint_path)
        except: pass

    start_time = time.time()
    stopped_epoch = initial_epoch
    if initial_epoch < EPOCHS:
        try:
            history = model.fit(train_ds, epochs=EPOCHS, initial_epoch=initial_epoch, validation_data=val_ds, callbacks=callbacks, verbose=1)
            stopped_epoch = initial_epoch + len(history.history['loss'])
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            
    training_time = time.time() - start_time
    return checkpoint_path, log_path, training_time, stopped_epoch


def generate_academic_report(model_func, model_name, num_classes, train_ds, val_ds, test_ds, class_names, checkpoint_path, log_path, training_time, stopped_epoch, norm_layer):
    print(f"\n--- Evaluation report: {model_name} ---")
    if not os.path.exists(checkpoint_path): return None

    model = model_func(num_classes, norm_layer)
    
    # Rebuild the exact partial-freeze layout used during fine-tuning
    # so that load_weights matches the trainable/frozen structure.
    if "FINETUNED" in model_name:
        model.trainable = True
        
        # freeze the first 70% of the base
        for layer in model.layers:
            if hasattr(layer, 'layers'): 
                freeze_limit = int(len(layer.layers) * 0.7)
                for i, sub_layer in enumerate(layer.layers):
                    if i < freeze_limit:
                        sub_layer.trainable = False
                    else:
                        sub_layer.trainable = True
        
        # keep BatchNormalization layers frozen
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                        sub_layer.trainable = False
            elif isinstance(layer, tf.keras.layers.BatchNormalization): 
                layer.trainable = False
    
    # with the layout matched, the weights now load cleanly
    model.load_weights(checkpoint_path)
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    
    y_true = []
    y_pred_probs = []
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        y_pred_probs.extend(model.predict(images, verbose=0))
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Save per-image predictions (for McNemar and any later metric/figure).
    try:
        _pred = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred_classes})
        for _ci, _cn in enumerate(class_names):
            _pred[f'prob_{_cn}'] = y_pred_probs[:, _ci]
        _pred.to_csv(os.path.join(output_dir, f"{model_name}_predictions.csv"), index=False)
    except Exception as _e:
        print(f"Prediction save error: {_e}")
    
    report_dict = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(output_dir, f"{model_name}_classification_report.csv"))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax[0])
    ax[0].set_title(f'{model_name} - Confusion Matrix')
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='OrRd', xticklabels=class_names, yticklabels=class_names, ax=ax[1])
    ax[1].set_title(f'{model_name} - Normalized')
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()
    
    # Training/validation history plot
    if os.path.exists(log_path):
        try:
            history_df = pd.read_csv(log_path)
            plt.figure(figsize=(14, 6))
            
            # Accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history_df['accuracy'], label='Train Accuracy', linewidth=2)
            plt.plot(history_df['val_accuracy'], label='Val Accuracy', linewidth=2, linestyle='--')
            plt.title(f'{model_name} - Accuracy over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            
            # Loss
            plt.subplot(1, 2, 2)
            plt.plot(history_df['loss'], label='Train Loss', linewidth=2)
            plt.plot(history_df['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
            plt.title(f'{model_name} - Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_training_history.png"))
            plt.close()
        except Exception as e:
            print(f"History plot error: {e}")

    # ROC curves (multiclass)
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    roc_auc = 0.0
    roc_auc_macro = 0.0
    try:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
        roc_auc = auc(fpr_micro, tpr_micro)

        # Macro-average AUC (mean of per-class AUCs)
        _pc = []
        for _i in range(num_classes):
            _f, _t, _ = roc_curve(y_true_bin[:, _i], y_pred_probs[:, _i]); _pc.append(auc(_f, _t))
        roc_auc_macro = float(np.nanmean(_pc))
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
        for i, color in zip(range(num_classes), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc_class = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, 
                     label=f'{class_names[i]} (AUC = {roc_auc_class:.2f})')
            
        plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', linewidth=4, 
                 label=f'Micro-average (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve (Multiclass)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"))
        plt.close()
        
    except Exception as e: 
        print(f"ROC plot error: {e}")
        roc_auc = 0.0
        roc_auc_macro = 0.0

    return {
        'Model': model_name,
        'Stopped Epoch': stopped_epoch,
        'Train Acc': train_acc,
        'Val Acc': val_acc,
        'Test Acc': test_acc,
        'Test F1 (Weighted)': report_dict['weighted avg']['f1-score'],
        'Test Recall (Weighted)': report_dict['weighted avg']['recall'],
        'Test AUC': roc_auc,
        'Test AUC (Macro)': roc_auc_macro,
        'Training Time (s)': training_time
    }

# Cross-model comparison plot
def plot_model_comparison(all_results_df, out_name="All_Models_Comparison.png"):
    """Plot bar charts comparing the models across the key metrics."""
    if all_results_df.empty:
        return
        
    metrics_to_plot = ['Test Acc', 'Test F1 (Weighted)', 'Test AUC', 'Training Time (s)']
    
    plt.figure(figsize=(18, 10))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        
        # bar plot
        sns.barplot(x='Model', y=metric, data=all_results_df, palette='viridis', hue='Model', legend=False)
        
        plt.title(f'Model Comparison: {metric}', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # annotate each bar with its value
        for index, row in all_results_df.iterrows():
            # value for this metric
            value = row[metric]
            # place the label just above the bar
            plt.text(index, value, f'{value:.3f}', color='black', ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, out_name))
    plt.close()
    print(f"\nComparison plot saved: {os.path.join(output_dir, out_name)}")

# ======================================================================
# CELL 4
# ======================================================================
# ============================ Fine-tuning ============================

def apply_fine_tuning(model_func, model_name, num_classes, train_ds, val_ds, test_ds, norm_layer, class_names):
    """Load stage-1 weights, unfreeze the top layers, and retrain with a low LR."""
    print(f"\n--- Fine-tuning: {model_name} ---")
    
    # File paths
    base_weights_path = os.path.join(output_dir, f"{model_name}_best_model.h5")
    ft_checkpoint_path = os.path.join(output_dir, f"{model_name}_FINETUNED_best_model.h5")
    ft_log_path = os.path.join(output_dir, f"{model_name}_FINETUNED_training_log.csv") # fine-tuning writes to its own log (separate from stage 1)
    
    # Check that the stage-1 weights exist
    if not os.path.exists(base_weights_path):
        print(f"ERROR: stage-1 weights not found for {model_name}.")
        print(f"   Expected path: {base_weights_path}")
        print("   Run stage 1 (feature extraction) first.")
        return None

    # Build the model and load stage-1 weights
    # The builder creates a frozen base; build first, then load weights.
    model = model_func(num_classes, norm_layer)
    print(f"Loading weights from '{base_weights_path}'...")
    model.load_weights(base_weights_path)

    # Partial unfreeze
    model.trainable = True
    
    # locate the nested base model
    for layer in model.layers:
        # nested base (transfer-learning model)
        if hasattr(layer, 'layers'): 
            # freeze the first 70%, keep the last 30% trainable
            freeze_limit = int(len(layer.layers) * 0.7)
            for i, sub_layer in enumerate(layer.layers):
                if i < freeze_limit:
                    sub_layer.trainable = False
                else:
                    sub_layer.trainable = True
    
    # keep BatchNormalization layers frozen
    for layer in model.layers:
        if hasattr(layer, 'layers'): # inside the nested base
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                    sub_layer.trainable = False
        elif isinstance(layer, tf.keras.layers.BatchNormalization): # any top-level BN we added
            layer.trainable = False
    
    print(f"Top layers unfrozen for fine-tuning.")

    # Recompile with a low learning rate
    ft_optimizer = AdamW(learning_rate=1e-5, weight_decay=1e-6)
    
    model.compile(optimizer=ft_optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        ModelCheckpoint(filepath=ft_checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        CSVLogger(ft_log_path, append=True), # appends to the fine-tuning log
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)
    ]

    # Resume epoch (from the fine-tuning log, if present)
    initial_epoch = 0
    if os.path.exists(ft_log_path):
        try:
            log_df = pd.read_csv(ft_log_path)
            if not log_df.empty:
                initial_epoch = log_df['epoch'].iloc[-1] + 1
                print(f"Resuming fine-tuning from epoch {initial_epoch}.")
        except: 
            pass
            
    # fine-tune for up to 200 additional epochs
    TOTAL_EPOCHS = initial_epoch + 200 

    # Train
    start_time = time.time()
    try:
        history = model.fit(train_ds, 
                            epochs=TOTAL_EPOCHS, 
                            initial_epoch=initial_epoch, 
                            validation_data=val_ds, 
                            callbacks=callbacks, 
                            verbose=1)
        stopped_epoch = initial_epoch + len(history.history['loss'])
    except KeyboardInterrupt:
        print("\nFine-tuning interrupted by user.")
        stopped_epoch = initial_epoch

    training_time = time.time() - start_time
    
    # Evaluate the fine-tuned model
    return generate_academic_report(model_func, f"{model_name}_FINETUNED", num_classes, 
                                    train_ds, val_ds, test_ds, class_names, 
                                    ft_checkpoint_path, ft_log_path, training_time, stopped_epoch, norm_layer)


# ============================ Automated multi-seed driver ============================

# Stage 1 (feature extraction): 3 from-scratch + 5 ImageNet-transfer models
PHASE1_MODELS = [
    (build_lenet5_scratch,         'LeNet-5'),
    (build_custom_modern_net_base, 'Custom_Modern_Net'),
    (build_resnet50_scratch,       'ResNet50-Scratch'),
    (build_vgg16,                  'VGG16'),
    (build_vgg19,                  'VGG19'),
    (build_resnet50,               'ResNet50'),
    (build_inceptionv3,            'InceptionV3'),
    (build_efficientnetb0,         'EfficientNetB0'),
]

# Stage 2 (fine-tuning): the two best transfer models
FINETUNE_MODELS = [
    (build_resnet50,       'ResNet50'),
    (build_efficientnetb0, 'EfficientNetB0'),
]

SUMMARY_COLS = ['Model', 'Stopped Epoch', 'Train Acc', 'Val Acc', 'Test Acc',
                'Test F1 (Weighted)', 'Test Recall (Weighted)',
                'Test AUC', 'Test AUC (Macro)', 'Training Time (s)']


def write_run_info(seed, phase, model_list):
    """Write a reproducibility-provenance JSON for each seed/stage."""
    import json as _json, platform, datetime
    info = {
        "seed": seed,
        "phase": phase,
        "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
        "tensorflow": tf.__version__,
        "python": platform.python_version(),
        "gpus": [g.name for g in tf.config.list_physical_devices('GPU')],
        "img_size": [IMG_HEIGHT, IMG_WIDTH],
        "batch_size": BATCH_SIZE,
        "epochs_phase1": EPOCHS,
        "use_head_l2": USE_HEAD_L2,
        "l2_value": 0.001,
        "optimizer": "AdamW",
        "lr_feature_extraction": 3e-4,
        "lr_fine_tuning": 1e-5,
        "models": [m[1] for m in model_list],
    }
    with open(os.path.join(output_dir, f"run_info_{phase}.json"), "w") as f:
        _json.dump(info, f, indent=2, ensure_ascii=False)


def save_summary(results, csv_name):
    if not results:
        return
    df = pd.DataFrame(results)
    cols = [c for c in SUMMARY_COLS if c in df.columns]
    df = df[cols]
    path = os.path.join(output_dir, csv_name)
    df.to_csv(path, index=False)
    print(f"\n--- {csv_name} ---")
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"Saved: {path}")
    tag = "PHASE1" if "PHASE1" in csv_name else "FINETUNED"
    try:
        plot_model_comparison(df, f"All_Models_Comparison_{tag}.png")
    except Exception:
        pass


def run_seed(seed):
    """Run one seed end to end: data -> stage 1 (FE) -> stage 2 (FT)."""
    global output_dir, USE_HEAD_L2
    set_seed(seed)
    output_dir = f"sonuclar_nisan2026_best_models_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 72)
    print(f"  SEED {seed}   ->   '{output_dir}'")
    print("=" * 72)

    # load data for this seed
    train_ds, val_ds, test_ds, class_names, num_classes = load_datasets()
    print("\nAdapting Z-score normalization on the training set...")
    normalization_layer = layers.Normalization(axis=-1)
    normalization_layer.adapt(train_ds.map(lambda x, y: x))
    print("Z-score layer ready.")

    # ----- Stage 1: feature extraction (head L2 off) -----
    USE_HEAD_L2 = False
    print(f"\nStage 1 (feature extraction) | head L2: off | {len(PHASE1_MODELS)} models")
    write_run_info(seed, "PHASE1", PHASE1_MODELS)
    p1 = []
    for mf, name in PHASE1_MODELS:
        keras.backend.clear_session()
        ckpt, log, t_time, stop_ep = train_robust_model(mf, name, num_classes, train_ds, val_ds, normalization_layer)
        m = generate_academic_report(mf, name, num_classes, train_ds, val_ds, test_ds,
                                     class_names, ckpt, log, t_time, stop_ep, normalization_layer)
        if m:
            p1.append(m)
    save_summary(p1, "tum_modeller_karsilastirma_zscore_PHASE1.csv")

    # ----- Stage 2: fine-tuning (head L2 on) -----
    set_seed(seed)          # reseed so fine-tuning is deterministic w.r.t. the seed
    USE_HEAD_L2 = True
    print(f"\nStage 2 (fine-tuning) | head L2: on (0.001) | {len(FINETUNE_MODELS)} models")
    write_run_info(seed, "FINETUNED", FINETUNE_MODELS)
    p2 = []
    for mf, name in FINETUNE_MODELS:
        keras.backend.clear_session()
        m = apply_fine_tuning(mf, name, num_classes, train_ds, val_ds, test_ds, normalization_layer, class_names)
        if m:
            p2.append(m)
    save_summary(p2, "tum_modeller_karsilastirma_zscore_FINETUNED.csv")
    print(f"\nSeed {seed} complete.")


def run_all():
    print(f"Starting automated run | seeds = {SEEDS}")
    for s in SEEDS:
        run_seed(s)
    print("\nAll seeds complete. Each seed folder holds the PHASE1 + FINETUNED CSVs, predictions, and run_info.")


if __name__ == "__main__":
    run_all()
