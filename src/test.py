print("Проверка окружения:")
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print("GPU доступен:", tf.config.list_physical_devices('GPU'))