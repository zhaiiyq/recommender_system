from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загружаем данные
@st.cache
def load_data():
    # Замените на ваш путь к датасету
    df = pd.read_csv('games1.csv')
    df['Название'] = df['Название'].fillna('')
    df['Описание'] = df['Описание'].fillna('')
    df['Жанры'] = df['Жанры'].fillna('')
    df['Цена'] = df['Цена'].fillna('')
    df['Разработчик'] = df['Разработчик'].fillna('')
    df['features'] = df['Описание'] + ' ' + df['Название'] + ' ' + df['Разработчик'] + ' ' + df['Жанры']
    return df

# Инициализация данных и модели
df = load_data()
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['features'])
indices = pd.Series(df.index, index=df['Название']).drop_duplicates()

# Функция поиска по запросу
def search_recommendation(query, top_n=10):
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices][['Название', 'Жанры','Разработчик', 'Общая оценка', 'Описание', 'Цена']]
    results['Сходство'] = cosine_sim[top_indices]
    return results

# Функция рекомендаций по товару
def recommend(title, num_recommendations=5):
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    game_indices = [i[0] for i in sim_scores]
    results = df.iloc[game_indices][['Название', 'Жанры', 'Разработчик', 'Общая оценка', 'Описание', 'Цена']]
    results['Сходство'] = [sim_scores[i][1] for i in range(len(sim_scores))]
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
    st.header("Рекомендации по товару")
    selected_title = st.selectbox("Выберите товар:", df['Название'].unique())
    if selected_title:
        recommendations = recommend(selected_title)
        st.write(recommendations)
