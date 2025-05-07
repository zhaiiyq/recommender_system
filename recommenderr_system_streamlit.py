import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загружаем данные
@st.cache_data
def load_data():
    # Замените на ваш путь к датасету
    df = pd.read_csv('games1.csv')
    df['Название'] = df['Название'].fillna('')
    df['Описание'] = df['Описание'].fillna('')
    df['Жанры'] = df['Жанры'].fillna('')
    df['Цена'] = df['Цена'].fillna('')
    df['Разработчик'] = df['Разработчик'].fillna('')

    # Взвешивание признаков: Название — важнее, Жанры — тоже весомые
    df['features'] = (
        df['Название'] * 3 + ' ' +
        df['Жанры'] * 2 + ' ' +
        df['Описание'] + ' ' +
        df['Разработчик']
    )
    return df

# Инициализация данных и модели
df = load_data()
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df['features'])

# Функция поиска по запросу
def search_recommendation(query, top_n=10):
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices][['Название', 'Жанры', 'Разработчик', 'Общая оценка', 'Описание', 'Цена']].copy()
    results['Сходство'] = cosine_sim[top_indices]
    return results

# Интерфейс Streamlit
st.title("Рекомендательная система по играм")

tab1, tab2 = st.tabs(["Поиск", "Рекомендации"])

with tab1:
    st.header("Поиск по запросу")
    query = st.text_input("Введите ваш запрос:")
    if query:
        results = search_recommendation(query)
        st.write(results)

with tab2:
    st.header("Рекомендации по игре")
    selected_title = st.selectbox("Выберите товар:", df['Название'].unique())
    if selected_title:
        recommendations = search_recommendation(selected_title)
        st.write(recommendations)
