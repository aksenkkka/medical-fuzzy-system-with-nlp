"""
medical review fuzzy logic analyzer

streamlit-застосунок із трьома вкладками:
1. ручний аналізатор — введення одного відгуку та отримання fuzzy-оцінки
2. файл з відгуками — завантаження csv, аналіз усіх відгуків і збереження до бд
3. рейтинг клінік — перегляд рейтингу клінік і деталізації відгуків
"""

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import func

from database.db import get_session, init_db
from database.models import Clinic, Review
from fuzzy_engine.inference import FuzzyEvaluator
from nlp_engine.semantic import NLPParser


# налаштування сторінки streamlit
st.set_page_config(page_title="Система аналізу відгуків", layout="wide")

# ініціалізація бази даних
init_db()


# кешоване завантаження nlp-парсера, щоб модель не завантажувалась повторно
@st.cache_resource()
def get_parser():
    return NLPParser().load()


# кешоване створення fuzzy-оцінювача
@st.cache_resource
def get_evaluator():
    return FuzzyEvaluator()


# створення основних об'єктів системи
parser = get_parser()
evaluator = get_evaluator()


# словник підписів для критеріїв оцінювання
DIM_LABELS = {
    "wait_time": "Час очікування",
    "doctor_comp": "Компетенція лікаря",
    "politeness": "Ввічливість",
    "cleanliness": "Чистота/Комфорт",
    "emotion": "Загальне враження",
    "consultation": "Враження від консультації",
}


def analyze_text(text: str) -> dict:
    # запуск nlp-модуля для вилучення ознак із тексту
    nlp_result = parser.extract_features(text)

    # отримання числових ознак за критеріями
    features = nlp_result["features"]

    # обчислення фінального fuzzy-рейтингу
    rating = evaluator.evaluate_from_features(features)

    # повернення результатів nlp-аналізу разом із fuzzy-рейтингом
    return {**nlp_result, "fuzzy_rating": rating}


def _gauge_color(value: float) -> str:
    # вибір кольору відповідно до значення рейтингу
    if value < 4:
        return "#dc3545"
    if value < 6.5:
        return "#ffc107"
    return "#28a745"


def gauge_chart(value: float, title: str = "Оцінка"):
    # побудова gauge-діаграми для візуалізації fuzzy-рейтингу
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": _gauge_color(value)},
                "steps": [
                    {"range": [0, 4], "color": "#ffe0e0"},
                    {"range": [4, 6.5], "color": "#fff3cd"},
                    {"range": [6.5, 10], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        )
    )

    # налаштування розміру та відступів діаграми
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=30, r=30))
    return fig


def get_or_create_clinic(session, name: str) -> Clinic:
    # пошук клініки за назвою у базі даних
    clinic = session.query(Clinic).filter_by(name=name).first()

    # якщо клініки ще немає, створюється новий запис
    if not clinic:
        clinic = Clinic(name=name)
        session.add(clinic)
        session.flush()

    return clinic


def save_review_to_db(session, clinic_name: str, text: str, result: dict, user_rating=None):
    # отримання або створення клініки
    clinic = get_or_create_clinic(session, clinic_name)

    # отримання оцінок за окремими критеріями
    features = result["features"]

    # створення запису відгуку для збереження в базі даних
    review = Review(
        clinic_id=clinic.id,
        original_text=text,
        user_rating=float(user_rating) if user_rating is not None else None,
        wait_time_score=features["wait_time"],
        doctor_comp_score=features["doctor_comp"],
        politeness_score=features["politeness"],
        cleanliness_score=features["cleanliness"],
        emotion_score=features["emotion"],
        consultation_score=features["consultation"],
        final_fuzzy_rating=result["fuzzy_rating"],
    )

    # додавання відгуку до поточної сесії
    session.add(review)


def read_uploaded_csv(uploaded_file):
    # зчитування завантаженого файлу у байтовому вигляді
    raw = uploaded_file.read()

    # визначення роздільника за першим рядком файлу
    first_line = raw[:1024].decode("utf-8-sig", errors="replace").splitlines()[0]
    sep = ";" if ";" in first_line else ","

    # спроба зчитати csv
    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            sep=sep,
            encoding="utf-8-sig",
            on_bad_lines="skip",
            engine="python",
        )

    # якщо перша спроба не спрацювала pandas  визначає роздільник
    except Exception:
        df = pd.read_csv(
            io.BytesIO(raw),
            sep=None,
            encoding="utf-8-sig",
            on_bad_lines="skip",
            engine="python",
        )

    # очищення назв колонок від зайвих символів і пробілів
    df.columns = [column.lstrip("\ufeff").strip() for column in df.columns]

    return df


# заголовок застосунку
st.title("Нечітка система оцінювання якості медичного обслуговування")

# створення трьох основних вкладок
tab1, tab2, tab3 = st.tabs(
    ["Ручний аналізатор", "Файл з відгуками", "Рейтинг клінік"]
)


# вкладка 1: ручний аналіз одного відгуку
with tab1:
    st.subheader("Ручний аналіз відгуку")
    st.caption(
        "Введіть відгук — система покаже знайдені ключові слова "
        "та оцінку нечіткої логіки"
    )

    # поле для введення тексту відгуку
    review_text = st.text_area(
        "Текст відгуку",
        height=120,
    )

    # запуск аналізу після натискання кнопки
    if st.button("Аналізувати", type="primary", key="btn_analyze"):
        if not review_text.strip():
            st.warning("Введіть текст відгуку.")
        else:
            # аналіз введеного тексту
            result = analyze_text(review_text)
            features = result["features"]
            found = result["found_keywords"]

            st.divider()

            # блок із загальною fuzzy-оцінкою
            col_gauge, col_info = st.columns([1, 2])

            with col_gauge:
                # виведення gauge-діаграми
                st.plotly_chart(
                    gauge_chart(result["fuzzy_rating"], "Fuzzy рейтинг"),
                    use_container_width=True,
                )

            with col_info:
                # виведення числового значення рейтингу
                st.metric("Fuzzy оцінка", f"{result['fuzzy_rating']} / 10")

                # виведення знайдених ключових слів за категоріями
                highlights = result["category_highlights"]
                st.markdown("**Знайдено слів:**")

                col_c, col_s = st.columns(2)

                with col_c:
                    words = highlights.get("competence", [])
                    st.markdown(
                        f"Компетенція: **{len(words)}** слів\n\n"
                        + (", ".join(f"`{word}`" for word in words) if words else "_нічого_")
                    )

                with col_s:
                    words = highlights.get("service", [])
                    st.markdown(
                        f"Сервіс: **{len(words)}** слів\n\n"
                        + (", ".join(f"`{word}`" for word in words) if words else "_нічого_")
                    )

            st.divider()
            st.subheader("Деталі по кожному критерію")

            # виведення оцінок за шістьма критеріями
            cols = st.columns(3)

            for index, (dim, label) in enumerate(DIM_LABELS.items()):
                with cols[index % 3]:
                    score = features[dim]
                    pos = found[dim]["positive"]
                    neg = found[dim]["negative"]
                    color = _gauge_color(score)

                    # назва критерію та його оцінка
                    st.markdown(
                        f"**{label}** — "
                        f"<span style='color:{color};font-size:1.1em'>{score:.1f}</span>/10",
                        unsafe_allow_html=True,
                    )

                    # позитивні знайдені ознаки
                    if pos:
                        st.caption("+ " + ", ".join(pos))

                    # негативні знайдені ознаки
                    if neg:
                        st.caption("- " + ", ".join(neg))

                    # повідомлення, якщо критерій залишився нейтральним
                    if not pos and not neg:
                        st.caption("_ключових слів не знайдено (нейтрально 5.0)_")

            st.divider()
            st.subheader("Графік якості критеріїв")

            # підготовка даних для радарної діаграми
            dims = list(DIM_LABELS.keys())
            labels = list(DIM_LABELS.values())
            values = [features[dim] for dim in dims]

            values_closed = values + [values[0]]
            labels_closed = labels + [labels[0]]

            # побудова радарної діаграми за критеріями
            fig_radar = go.Figure(
                go.Scatterpolar(
                    r=values_closed,
                    theta=labels_closed,
                    fill="toself",
                    line_color="#0d6efd",
                )
            )

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=False,
                height=380,
            )

            st.plotly_chart(fig_radar, use_container_width=True)


# вкладка 2: завантаження csv-файлу з відгуками
with tab2:
    st.subheader("Завантаження файлу з відгуками")
    st.caption(
        "Завантажте CSV файл з колонками: **name** (клініка), "
        "**review_text** (текст відгуку). "
        "Необов'язкові колонки — **review_rating** та **review_datetime**"
    )

    # завантаження csv-файлу користувачем
    uploaded_file = st.file_uploader("Оберіть CSV файл", type=["csv"])

    if uploaded_file:
        try:
            # зчитування csv-файлу
            df = read_uploaded_csv(uploaded_file)
        except Exception as error:
            st.error(f"Помилка читання файлу: {error}")
            df = None

        if df is not None:
            # виведення знайдених колонок
            st.write("Знайдені колонки:", list(df.columns))

            # перевірка наявності обов'язкових колонок
            if "name" not in df.columns or "review_text" not in df.columns:
                st.error("Файл повинен містити колонки: **name**, **review_text**.")
            else:
                st.success(f"Файл завантажено: {len(df)} рядків")

                # показ перших п'яти рядків файлу
                st.dataframe(df.head(5), use_container_width=True)

                col_opt1, col_opt2 = st.columns(2)

                with col_opt1:
                    # опція пропуску надто коротких відгуків
                    skip_short = st.checkbox(
                        "Пропускати рядки < 5 символів",
                        value=True,
                    )

                with col_opt2:
                    # опція збереження результатів до бази даних
                    save_to_db = st.checkbox(
                        "Зберігати результати до БД",
                        value=True,
                    )

                # запуск пакетного аналізу
                if st.button("Запустити аналіз", type="primary", key="btn_process"):
                    results = []
                    session = get_session()
                    progress = st.progress(0, text="Аналізую відгуки...")

                    # обробка кожного рядка з файлу
                    for index, row in df.iterrows():
                        text = str(row.get("review_text", "")).strip()
                        clinic_name = str(row.get("name", "Невідома клініка")).strip()
                        user_rating = row.get("review_rating", None)

                        # пропуск коротких або порожніх текстів
                        if skip_short and len(text) < 5:
                            continue
                        if text.lower() == "nan":
                            continue

                        # аналіз одного відгуку
                        result = analyze_text(text)

                        # формування рядка результату для таблиці
                        row_result = {
                            "Клініка": clinic_name,
                            "Відгук": text[:80] + ("..." if len(text) > 80 else ""),
                            "Fuzzy оцінка": result["fuzzy_rating"],
                            "Оцінка користувача": user_rating,
                            "Очікування": result["features"]["wait_time"],
                            "Компетенція": result["features"]["doctor_comp"],
                            "Ввічливість": result["features"]["politeness"],
                            "Чистота": result["features"]["cleanliness"],
                            "Враження": result["features"]["emotion"],
                            "Консультація": result["features"]["consultation"],
                        }

                        results.append(row_result)

                        # збереження результату аналізу до бази даних
                        if save_to_db:
                            save_review_to_db(
                                session,
                                clinic_name,
                                text,
                                result,
                                user_rating,
                            )

                        progress.progress(
                            min((index + 1) / len(df), 1.0),
                            text=f"Аналіз відгуків... {index + 1}/{len(df)}",
                        )

                    # підтвердження змін у базі даних
                    if save_to_db:
                        session.commit()

                    session.close()
                    progress.empty()

                    if results:
                        # створення таблиці результатів
                        result_df = pd.DataFrame(results)

                        st.success(f"Оброблено: {len(result_df)} відгуків")

                        # виведення узагальнених метрик
                        col_mean_fuzzy, col_mean_user = st.columns(2)

                        with col_mean_fuzzy:
                            st.metric(
                                "Середня fuzzy оцінка",
                                f"{result_df['Fuzzy оцінка'].mean():.2f} / 10",
                            )

                        with col_mean_user:
                            if result_df["Оцінка користувача"].notna().any():
                                user_mean = pd.to_numeric(
                                    result_df["Оцінка користувача"],
                                    errors="coerce",
                                ).mean()

                                st.metric(
                                    "Середня оцінка користувачів",
                                    f"{user_mean:.2f}",
                                )

                        # таблиця з результатами аналізу
                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        # побудова гістограми розподілу fuzzy-оцінок
                        fig_dist = px.histogram(
                            result_df,
                            x="Fuzzy оцінка",
                            nbins=20,
                            title="Розподіл fuzzy оцінок",
                            labels={
                                "Fuzzy оцінка": "Оцінка (0-10)",
                                "count": "Кількість",
                            },
                        )

                        st.plotly_chart(fig_dist, use_container_width=True)

                        # формування csv-файлу з результатами для завантаження
                        csv_out = result_df.to_csv(index=False).encode("utf-8")

                        st.download_button(
                            "Завантажити результати CSV",
                            data=csv_out,
                            file_name="analyzed_reviews.csv",
                            mime="text/csv",
                        )

                    else:
                        st.warning("Не знайдено рядків для обробки.")


# вкладка 3: рейтинг клінік на основі збережених результатів
with tab3:
    st.subheader("Рейтинг клінік")
    st.caption(
        "Дані з бази. Рейтинг є середнім арифметичним fuzzy-оцінок "
        "усіх відгуків однієї клініки"
    )

    col_refresh, col_search = st.columns([1, 3])

    with col_refresh:
        # оновлення сторінки з рейтингом
        if st.button("Оновити", key="btn_refresh"):
            st.rerun()

    with col_search:
        # поле пошуку клініки за назвою
        search_query = st.text_input(
            "Пошук клініки",
            placeholder="Введіть назву...",
            label_visibility="collapsed",
        )

    session = get_session()

    # завантаження клінік із кількістю відгуків і середнім fuzzy-рейтингом
    rows = (
        session.query(
            Clinic.id,
            Clinic.name,
            func.count(Review.id).label("review_count"),
            func.avg(Review.final_fuzzy_rating).label("avg_rating"),
        )
        .outerjoin(Review, Review.clinic_id == Clinic.id)
        .group_by(Clinic.id)
        .order_by(func.avg(Review.final_fuzzy_rating).desc())
        .all()
    )

    session.close()

    if not rows:
        st.info("База порожня. Завантажте відгуки на вкладці 'Файл з відгуками'.")
    else:
        # фільтрація клінік за пошуковим запитом
        if search_query:
            rows = [
                row for row in rows
                if search_query.lower() in row.name.lower()
            ]

        if not rows:
            st.warning("Клінік не знайдено.")
        else:
            # підготовка даних для стовпчикової діаграми рейтингу клінік
            clinic_names = [
                row.name for row in rows
                if row.avg_rating is not None
            ]

            clinic_ratings = [
                round(row.avg_rating, 2) for row in rows
                if row.avg_rating is not None
            ]

            # побудова рейтингу клінік у вигляді горизонтальної діаграми
            if clinic_ratings:
                fig_bar = px.bar(
                    x=clinic_ratings,
                    y=clinic_names,
                    orientation="h",
                    labels={
                        "x": "Fuzzy рейтинг (0-10)",
                        "y": "Клініка",
                    },
                    title="Рейтинг клінік",
                    color=clinic_ratings,
                    color_continuous_scale=["#dc3545", "#ffc107", "#28a745"],
                    range_color=[0, 10],
                )

                fig_bar.update_layout(
                    height=max(300, len(clinic_names) * 40),
                    showlegend=False,
                    coloraxis_showscale=False,
                )

                st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()

            # виведення списку клінік із середнім рейтингом
            for row in rows:
                avg = f"{row.avg_rating:.1f}" if row.avg_rating is not None else "—"
                count = row.review_count
                rating_val = row.avg_rating or 0
                color = _gauge_color(rating_val)

                col_name, col_rating, col_btn = st.columns([4, 2, 1])

                with col_name:
                    st.markdown(f"**{row.name}**  `{count} відгуків`")

                with col_rating:
                    st.markdown(
                        f"<span style='color:{color};font-size:1.3em;font-weight:bold'>"
                        f"{avg}</span> / 10",
                        unsafe_allow_html=True,
                    )

                with col_btn:
                    # вибір клініки для перегляду її відгуків
                    if st.button("Відгуки", key=f"clinic_{row.id}"):
                        st.session_state["selected_clinic_id"] = row.id
                        st.session_state["selected_clinic_name"] = row.name

            # отримання обраної клініки зі стану сторінки
            selected_id = st.session_state.get("selected_clinic_id")

            if selected_id:
                selected_name = st.session_state.get("selected_clinic_name", "")

                st.divider()
                st.subheader(f"Відгуки: {selected_name}")

                # закриття блоку деталізації клініки
                if st.button("Закрити", key="btn_close_clinic"):
                    del st.session_state["selected_clinic_id"]
                    del st.session_state["selected_clinic_name"]
                    st.rerun()

                session2 = get_session()

                # завантаження відгуків обраної клініки
                reviews = (
                    session2.query(Review)
                    .filter_by(clinic_id=selected_id)
                    .order_by(Review.final_fuzzy_rating.desc())
                    .all()
                )

                session2.close()

                if not reviews:
                    st.info("Відгуків поки немає.")
                else:
                    # виведення кожного відгуку з деталізацією оцінок
                    for review in reviews:
                        with st.expander(
                            f"[{review.final_fuzzy_rating:.1f}/10] "
                            f"{review.original_text[:80]}"
                            f"{'...' if len(review.original_text) > 80 else ''}"
                        ):
                            st.write(review.original_text)

                            cols = st.columns(6)

                            labels = [
                                ("Очікування", review.wait_time_score),
                                ("Компетенція", review.doctor_comp_score),
                                ("Ввічливість", review.politeness_score),
                                ("Чистота", review.cleanliness_score),
                                ("Враження", review.emotion_score),
                                ("Консультація", review.consultation_score),
                            ]

                            # виведення оцінок за всіма критеріями
                            for col, (label, value) in zip(cols, labels):
                                with col:
                                    color = _gauge_color(value or 5)
                                    st.markdown(
                                        f"**{label}**\n\n"
                                        f"<span style='color:{color}'>{value:.1f}</span>",
                                        unsafe_allow_html=True,
                                    )

                            # виведення користувацької оцінки, якщо вона була у файлі
                            if review.user_rating is not None:
                                st.caption(
                                    f"Оцінка користувача: {review.user_rating}/5"
                                )