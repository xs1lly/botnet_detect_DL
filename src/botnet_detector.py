
# -*- coding: utf-8 -*-
"""
Система обнаружения ботнетов на основе LSTM
С улучшенной обработкой признаков (без src/dst IP)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from scapy.all import rdpcap
import os
import matplotlib.pyplot as plt
import logging

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_pcap(file_path: str, max_packets: int = 500) -> pd.DataFrame:
    packets = rdpcap(file_path)[:max_packets]
    data = []

    for pkt in packets:
        if 'IP' in pkt:
            row = [
                float(pkt.time),
                len(pkt),
                pkt.sport if 'TCP' in pkt else 0,
                pkt.dport if 'TCP' in pkt else 0,
                1 if 'TCP' in pkt else (2 if 'UDP' in pkt else 0)
            ]
            data.append(row)

    return pd.DataFrame(data, columns=['timestamp', 'size', 'sport', 'dport', 'proto'])


def prepare_dataset(data_dir: str, seq_length: int = 30) -> tuple:
    X, y = [], []

    for file in os.listdir(data_dir):
        if file.endswith('.pcap'):
            label = 1 if 'botnet' in file else 0
            logger.info(f"Обработка {file} (метка: {label})")

            try:
                df = process_pcap(os.path.join(data_dir, file))
                if len(df) < seq_length:
                    continue

                for i in range(0, len(df) - seq_length, seq_length // 2):
                    seq = df.iloc[i:i + seq_length].values
                    X.append(seq)
                    y.append(label)
            except Exception as e:
                logger.error(f"Ошибка обработки {file}: {str(e)}")

    return np.array(X), np.array(y)


def build_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, name='LSTM1'),
        Dropout(0.2),
        LSTM(32, name='LSTM2'),
        Dense(16, activation='relu', name='Dense1'),
        Dense(1, activation='sigmoid', name='Output')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model


def plot_results(history, model, X_test, y_test):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность (обучение)')
    plt.plot(history.history['val_accuracy'], label='Точность (валидация)')
    plt.title('Динамика точности')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Потери (обучение)')
    plt.plot(history.history['val_loss'], label='Потери (валидация)')
    plt.title('Динамика потерь')
    plt.legend()
    plt.show()

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Нормальный', 'Ботнет'])
    disp.plot()
    plt.title('Матрица ошибок')
    plt.show()


if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("Загрузка и подготовка данных...")
    X, y = prepare_dataset(DATA_DIR)

    if len(X) == 0:
        raise ValueError("Не найдено PCAP-файлов в папке data!")

    unique, counts = np.unique(y, return_counts=True)
    print("Распределение классов:", dict(zip(unique, counts)))

    if abs(counts[0] - counts[1]) > 10:
        min_class = min(counts)
        idx_0 = np.where(y == 0)[0][:min_class]
        idx_1 = np.where(y == 1)[0][:min_class]
        idx_balanced = np.concatenate([idx_0, idx_1])
        np.random.shuffle(idx_balanced)
        X = X[idx_balanced]
        y = y[idx_balanced]
        print("Сбалансировано:", np.bincount(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    logger.info("Создание модели...")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    logger.info("Обучение модели...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=1
    )

    logger.info("Оценка модели...")
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)

    print("\nРезультаты:")
    print(f"Точность: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-мера: {f1:.2%}")

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(recalls, precisions, marker='.')
    plt.title("Precision-Recall кривая (F1 визуализация)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.show()

    plot_results(history, model, X_test, y_test)

    model.save("botnet_model.h5")
    logger.info("Модель сохранена в botnet_model.h5")
