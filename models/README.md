# 📦 Models Directory

Ця директорія містить збережені ваги найкращої навченої нейромережі та артефакти препроцесингу для продакшн-інференсу.

## 🎯 Найкраща архітектура: Bidirectional LSTM
* **Файл моделі:** `best_model.keras`
* **Файл токенізатора:** `tokenizer.pkl` (словник на 15 000 найчастіших токенів)
* **Параметри послідовності:** `MAX_LEN = 200`, `padding='post'`, `truncating='post'`

---

## 📊 Результати на тестовій вибірці (Test Set Benchmarking):
| Метрика | Значення |
| :--- | :--- |
| **Accuracy** | **90.75%** |
| **Precision** | **90.60%** |
| **Recall** | **90.90%** |
| **F1-Score** | **90.75%** |
| **ROC-AUC** | **0.9682** |
| **Розмір моделі** | ~11.8 MB (< 100 MB, відповідає лімітам Git) |

---

## 🛠️ Як завантажити та використовувати для інференсу:
```python
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Завантаження токенізатора
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# 2. Завантаження навченої моделі
model = tf.keras.models.load_model('models/best_model.keras')

# 3. Передбачення тональності
def predict(text):
    seq = tokenizer.texts_to_sequences([text.lower()])
    padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    proba = float(model.predict(padded)[0][0])
    label = 'Positive' if proba >= 0.5 else 'Negative'
    return {'sentiment': label, 'confidence': proba}
```
