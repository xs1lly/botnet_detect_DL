# -*- coding: utf-8 -*-
"""
Система обнаружения ботнетов на основе LSTM
Адаптировано под CTU-13 (Scenario 1 и 13)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scapy.all import rdpcap
import os
import matplotlib.pyplot as plt
import logging

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Модуль обработки PCAP-файлов ---
def process_pcap(file_path: str, max_packets: int = 500) -> pd.DataFrame:
    """
    Извлекает признаки из PCAP-файла.
    Возвращает DataFrame с колонками:
    ['timestamp', 'size', 'src_ip', 'dst_ip', 'sport', 'dport', 'proto']
    """
    packets = rdpcap(file_path)[:max_packets]
    data = []
    
    for pkt in packets:
        if 'IP' in pkt:
            row = [
                float(pkt.time),  # Временная метка
                len(pkt),        # Размер пакета
                int(pkt['IP'].src.split('.')[-1]),  # Последний октет IP источника
                int(pkt['IP'].dst.split('.')[-1]),  # Последний октет IP назначения
                pkt.sport if 'TCP' in pkt else 0,   # Порт источника
                pkt.dport if 'TCP' in pkt else 0,   # Порт назначения
                1 if 'TCP' in pkt else (2 if 'UDP' in pkt else 0)  # Протокол
            ]
            data.append(row)
    
    return pd.DataFrame(data, columns=['timestamp', 'size', 'src_ip', 'dst_ip', 'sport', 'dport', 'proto'])

# --- 2. Подготовка датасета ---
def prepare_dataset(data_dir: str, seq_length: int = 30) -> tuple:
    """
    Загружает PCAP-файлы и создает последовательности для LSTM.
    Возвращает:
    - X (np.array): данные формы [samples, seq_length, features]
    - y (np.array): метки (0=нормальный трафик, 1=ботнет)
    """
    X, y = [], []
    
    for file in os.listdir(data_dir):
        if file.endswith('.pcap'):
            # Определяем метку по имени файла
            label = 1 if file == 'botnet.pcap' else 0  # Четкое разделение по именам
            logger.info(f"Обработка {file} (метка: {label})")
            
            try:
                df = process_pcap(os.path.join(data_dir, file))
                if len(df) < seq_length:
                    continue
                
                # Создаем последовательности
                for i in range(0, len(df) - seq_length, seq_length // 2):
                    seq = df.iloc[i:i + seq_length].values
                    X.append(seq)
                    y.append(label)
            except Exception as e:
                logger.error(f"Ошибка обработки {file}: {str(e)}")
    
    return np.array(X), np.array(y)

# --- 3. Модель LSTM ---
def build_model(input_shape: tuple) -> Sequential:
    """Создает модель LSTM для классификации."""
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

# --- 4. Визуализация результатов ---
def plot_results(history, model, X_test, y_test):
    """Рисует графики обучения и матрицу ошибок."""
    plt.figure(figsize=(12, 4))
    
    # Графики точности и потерь
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
    
    # Матрица ошибок
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Нормальный', 'Ботнет'])
    disp.plot()
    plt.title('Матрица ошибок')
    plt.show()

# --- Основной код ---
if __name__ == "__main__":
    # Путь к данным
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..","data")  # Правильный относительный путь    os.makedirs(DATA_DIR, exist_ok=True)

    # Загрузка и подготовка данных
    logger.info("Загрузка данных...")
    X, y = prepare_dataset(DATA_DIR)
    
    if len(X) == 0:
        raise ValueError("Не найдено PCAP-файлов в папке data!")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Нормализация
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Создание модели
    logger.info("Создание модели...")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    # Обучение
    logger.info("Обучение модели...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=1
    )
    
    # Оценка
    logger.info("Оценка модели...")
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nРезультаты:")
    print(f"Точность: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    
    # Визуализация
    plot_results(history, model, X_test, y_test)
    
    # Сохранение модели
    model.save("botnet_model.h5")
    logger.info("Модель сохранена в botnet_model.h5")