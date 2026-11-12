# 🎬 Movie Review Sentiment Analysis (IMDb 50k)

Проєкт з дисципліни **«Глибинне навчання» (ІПЗ-4)**.  
Розробка інтелектуального сервісу на базі глибоких нейронних мереж (DNN / LSTM) для класифікації тональності відгуків глядачів на фільми з інтерактивним веб-інтерфейсом на **Streamlit**.

---

## 👥 Команда проєкту
* **Денис** — *Data & Deployment Lead* (EDA, пайплайни препроцесингу, архітектура Streamlit-додатку, деплоймент).
* **Мирослав** — *ML/DL Engineer & Presentation Storyteller* (архітектура та навчання нейромереж, оцінка метрик, інференс, презентація).

---

## 📌 Датасет
Використовується датасет **IMDb Dataset of 50K Movie Reviews**:
* Джерело: [Kaggle - IMDb Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* Розмір: 50 000 текстових відгуків (збалансований: 25k позитивних, 25k негативних).

---

## 📂 Структура проєкту
```text
movie-review-sentiment-analysis/
├── data/                  # Датасет (локально, ігнорується git)
├── models/                # Збережені ваги та токенізатори
├── notebooks/             # Дослідницькі Jupyter-ноутбуки
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── presentation/          # Матеріали презентації для нетехнічної аудиторії
├── src/                   # Модульні скрипти для навчання та інференсу
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── app.py                 # Streamlit веб-додаток
├── requirements.txt       # Список залежностей
└── README.md              # Документація проєкту
```

---

## 🚀 Швидкий старт

### 1. Клонування репозиторію
```bash
git clone https://github.com/<username>/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis
```

### 2. Створення та активація віртуального середовища
```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 3. Встановлення залежностей
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 План розробки (3 тижні)
- [x] **Тиждень 1:** Налаштування репозиторію, GitHub Projects, розвідувальний аналіз даних (EDA), препроцесинг, перший бейзлайн.
- [ ] **Тиждень 2:** Побудова моделей глибокого навчання (LSTM/BiLSTM), модульні скрипти `src/`, запуск прототипу на Streamlit.
- [ ] **Тиждень 3:** Покращення інтерфейсу, тестування, оформлення документації та підготовка презентації для нетехнічної аудиторії.
