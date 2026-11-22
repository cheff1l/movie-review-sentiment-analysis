"""
Interactive Web Application for Movie Review Sentiment Analysis.
Framework: Streamlit
Author: Denis (Data & Deployment Lead)
Model Architecture: Bidirectional LSTM (by Miroslav)
"""

import os
import sys
import time
from typing import Dict, Any, List
import pandas as pd
import streamlit as st

# Add current directory to path to support modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import TextPreprocessor
from src.tokenizer import ReviewTokenizer

# Page configuration
st.set_page_config(
    page_title="Movie Sentiment AI | Рецензії на фільми",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern, polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #FF8F00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E1E24;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #33333F;
        text-align: center;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(46, 125, 50, 0.25));
        border: 1px solid #4CAF50;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.15), rgba(198, 40, 40, 0.25));
        border: 1px solid #F44336;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .tag-pos {
        background-color: rgba(76, 175, 80, 0.25);
        color: #81C784;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        margin: 2px;
        display: inline-block;
    }
    .tag-neg {
        background-color: rgba(244, 67, 54, 0.25);
        color: #E57373;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_components():
    """Cached loader for ML models and tokenizer artifacts."""
    model_path = os.path.join("models", "best_model.keras")
    tokenizer_path = os.path.join("models", "tokenizer.pkl")

    has_artifacts = os.path.exists(model_path) and os.path.exists(tokenizer_path)
    
    preprocessor = TextPreprocessor()
    tokenizer = None
    model = None

    if has_artifacts:
        try:
            from src.predict import get_inference_artifacts
            model, tokenizer, preprocessor = get_inference_artifacts(model_path, tokenizer_path)
        except Exception as e:
            st.sidebar.warning(f"Помилка завантаження моделі Keras: {e}. Працює евристичний бейзлайн.")
            has_artifacts = False

    return preprocessor, tokenizer, model, has_artifacts


def heuristic_predict(text: str, preprocessor: TextPreprocessor) -> Dict[str, Any]:
    """Fallback heuristic classifier if DL model weights are missing or running in lightweight mode."""
    cleaned = preprocessor.clean_text(text)
    pos_words = {"masterpiece", "brilliant", "excellent", "superb", "fantastic", "amazing", "wonderful", "great", "loved", "enjoyed", "good", "best", "stunning", "perfect"}
    neg_words = {"terrible", "horrible", "awful", "worst", "waste", "boring", "poor", "bad", "stupid", "annoying", "mess", "unwatchable", "fail", "hate"}
    
    words = cleaned.split()
    pos_count = sum(1 for w in words if w in pos_words)
    neg_count = sum(1 for w in words if w in neg_words)

    # Simple sigmoid-like confidence approximation
    diff = pos_count - neg_count
    if "not" in words or "never" in words or "no" in words:
        diff -= 1

    if diff > 0:
        label = "Positive"
        conf = min(50.0 + (pos_count * 15.0), 96.5)
        prob = conf / 100.0
    elif diff < 0:
        label = "Negative"
        conf = min(50.0 + (neg_count * 15.0), 96.5)
        prob = 1.0 - (conf / 100.0)
    else:
        label = "Positive" if len(words) % 2 == 0 else "Negative"
        conf = 52.0
        prob = 0.52 if label == "Positive" else 0.48

    highlights = []
    for w in words:
        if w in pos_words:
            highlights.append({"word": w, "sentiment": "positive"})
        elif w in neg_words:
            highlights.append({"word": w, "sentiment": "negative"})

    return {
        "sentiment": label,
        "confidence": conf,
        "probability": prob,
        "cleaned_text": cleaned,
        "word_highlights": highlights
    }


def analyze_text(text: str, preprocessor, tokenizer, model, has_artifacts: bool) -> Dict[str, Any]:
    """Routing prediction through Deep Learning BiLSTM model or fallback."""
    if has_artifacts and model is not None and tokenizer is not None:
        from src.predict import predict_sentiment
        return predict_sentiment(text)
    else:
        return heuristic_predict(text, preprocessor)


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/movie-projector.png", width=80)
    st.title("🎬 Movie Sentiment AI")
    st.markdown("**Дисципліна:** Глибинне навчання (ІПЗ-4)")
    st.markdown("**Датасет:** [IMDb 50k Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)")
    st.markdown("---")
    
    st.markdown("### 👥 Команда:")
    st.markdown("👨‍💻 **Денис** — *Data & Deployment Lead*")
    st.markdown("🧠 **Мирослав** — *ML/DL Engineer*")
    
    st.markdown("---")
    st.markdown("### ⚙️ Архітектура:")
    st.markdown("* **Модель:** Bidirectional LSTM")
    st.markdown("* **Словник:** 15,000 слів")
    st.markdown("* **F1-Score:** `90.75%`")
    st.markdown("* **Розмір:** ~11.8 MB")
    
    st.markdown("---")
    mode = st.radio("📌 Оберіть режим:", ["Експрес-аналіз відгуку", "Пакетний бізнес-аналіз (CSV)"])


# --- MAIN CONTENT ---
st.markdown('<div class="main-header">🎬 AI-Аналізатор Тональності Відгуків на Фільми</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Швидка класифікація глядацьких емоцій за допомогою глибинних нейромереж (BiLSTM)</div>', unsafe_allow_html=True)

preprocessor, tokenizer, model, has_artifacts = load_components()

if not has_artifacts:
    st.info("ℹ️ Працює режим швидкого прототипування. Для повної BiLSTM нейромережі ваги завантажуються з `models/best_model.keras`.")


# --- MODE 1: SINGLE REVIEW ANALYSIS ---
if mode == "Експрес-аналіз відгуку":
    st.subheader("📝 Введіть або вставте текст відгуку глядача:")

    # Quick example buttons
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    preset_text = ""
    if col_ex1.button("Приклад 1 (Позитивний) 🟢"):
        preset_text = "This movie is an absolute masterpiece! Brilliant acting by the lead cast and stunning cinematography. Highly recommended!"
    elif col_ex2.button("Приклад 2 (Негативний) 🔴"):
        preset_text = "Total waste of time. The plot was completely boring, terrible directing and ridiculous dialogue. Worst film of the year."
    elif col_ex3.button("Приклад 3 (Змішаний/Іронічний) 🟡"):
        preset_text = "The special effects were not bad at all, but the script was horrible and the characters were stupid."

    user_review = st.text_area(
        "Текст рецензії (англійською мовою):",
        value=preset_text if preset_text else "",
        height=140,
        placeholder="Наприклад: I truly loved this film! The storyline was mesmerizing and emotional..."
    )

    col_btn, col_clean = st.columns([1, 4])
    analyze_clicked = col_btn.button("🔍 Аналізувати тональність", type="primary", use_container_width=True)

    if analyze_clicked or preset_text:
        if not user_review.strip():
            st.warning("⚠️ Будь ласка, введіть текст для аналізу!")
        else:
            with st.spinner("Нейромережа обробляє текст..."):
                time.sleep(0.2)
                res = analyze_text(user_review, preprocessor, tokenizer, model, has_artifacts)

            st.markdown("---")
            st.subheader("📊 Результат класифікації:")

            col1, col2, col3 = st.columns([2, 2, 3])

            with col1:
                if res["sentiment"] == "Positive":
                    st.markdown("""
                    <div class="sentiment-positive">
                        <h2 style="color: #4CAF50; margin: 0;">🟢 ПОЗИТИВНИЙ</h2>
                        <p style="margin: 5px 0 0 0; color: #E0E0E0;">Глядач рекомендує цей фільм</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="sentiment-negative">
                        <h2 style="color: #F44336; margin: 0;">🔴 НЕГАТИВНИЙ</h2>
                        <p style="margin: 5px 0 0 0; color: #E0E0E0;">Глядач незадоволений переглядом</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.metric(
                    label="Впевненість нейромережі:",
                    value=f"{res['confidence']:.1f}%",
                    delta=f"{res['probability']:.4f} prob"
                )
                st.progress(res["confidence"] / 100.0)

            with col3:
                st.markdown("**Ключові емоційні маркери, знайдені в тексті:**")
                highlights = res.get("word_highlights", [])
                if highlights:
                    html_tags = ""
                    for h in highlights:
                        cls = "tag-pos" if h["sentiment"] == "positive" else "tag-neg"
                        html_tags += f'<span class="{cls}">{h["word"]}</span> '
                    st.markdown(html_tags, unsafe_allow_html=True)
                else:
                    st.write("Специфічних полярних маркерів не виявлено (контекстне передбачення).")

            with st.expander("🔍 Деталі пайплайну обробки (для розробників):"):
                st.write("**Очищений та нормалізований текст:**")
                st.code(res.get("cleaned_text", ""))
                st.write(f"**Ймовірність сигмоїди:** `{res.get('probability', 0.5):.5f}`")


# --- MODE 2: BATCH CSV ANALYSIS ---
else:
    st.subheader("📁 Пакетний аналіз відгуків для кінотеатрів та стрімінгів")
    st.markdown("Завантажте `.csv` файл із колонкою `review` для миттєвої генерації бізнес-звіту:")

    uploaded_file = st.file_uploader("Оберіть CSV файл:", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if "review" not in df_upload.columns:
                st.error("❌ Завантажений файл повинен містити стовпчик з назвою 'review'!")
            else:
                st.success(f"✅ Успішно завантажено {len(df_upload)} відгуків!")
                sample_limit = min(len(df_upload), 250)
                st.caption(f"Для швидкої онлайн-демонстрації аналізуються перші {sample_limit} рядків.")

                if st.button("🚀 Запустити пакетну аналітику", type="primary"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for idx, text in enumerate(df_upload["review"].iloc[:sample_limit]):
                        res = analyze_text(str(text), preprocessor, tokenizer, model, has_artifacts)
                        results.append(res["sentiment"])
                        progress_bar.progress((idx + 1) / sample_limit)

                    df_res = df_upload.iloc[:sample_limit].copy()
                    df_res["Predicted_Sentiment"] = results

                    st.markdown("---")
                    st.subheader("📈 Підсумкова аналітика аудиторії:")

                    cnt = pd.Series(results).value_counts()
                    pos_share = (cnt.get("Positive", 0) / len(results)) * 100
                    neg_share = (cnt.get("Negative", 0) / len(results)) * 100

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Загалом проаналізовано", f"{len(results)} відгуків")
                    m2.metric("Позитивні відгуки", f"{pos_share:.1f}%", delta="Задоволені")
                    m3.metric("Негативні відгуки", f"{neg_share:.1f}%", delta="-Незадоволені", delta_color="inverse")

                    st.bar_chart(cnt)

                    st.markdown("### Таблиця результатів:")
                    st.dataframe(df_res[["review", "Predicted_Sentiment"]].head(20), use_container_width=True)

                    csv_export = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Завантажити повний звіт у CSV",
                        data=csv_export,
                        file_name="sentiment_analysis_report.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Помилка при читанні файлу: {e}")
