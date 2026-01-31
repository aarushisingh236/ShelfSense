import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="📚 Book Decision Dashboard",
    layout="wide"
)

CSV_PATH = Path("Book_List_Cleaned.csv")

# =====================================================
# DATA HELPERS
# =====================================================
def load_books():
    return pd.read_csv(CSV_PATH)


def save_books(df):
    df.to_csv(CSV_PATH, index=False)


def rating_category(r):
    if r >= 4.5:
        return "Excellent"
    elif r >= 3.5:
        return "Good"
    else:
        return "Average"


def pages_category(p):
    if p <= 150:
        return "Short"
    elif p <= 300:
        return "Medium"
    elif p <= 500:
        return "Long"
    else:
        return "Very Long"


def priority_weight(p):
    return {"High": 1.0, "Medium": 0.7, "Low": 0.4}[p]

# =====================================================
# SHARED BOOK FORM
# =====================================================
def book_form(book=None):
    is_edit = book is not None

    st.subheader("📖 Book Details")

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Title *", book["title"] if is_edit else "", placeholder="The Metamorphosis")
        author = st.text_input("Author *", book["author"] if is_edit else "", placeholder="Franz Kafka")
        genre = st.text_input("Genre", book["genre"] if is_edit else "", placeholder="Existential, Absurdist Fiction")
        book_type = st.text_input("Type", book["type"] if is_edit else "", placeholder="Fiction (Novella)")
        language = st.text_input("Language", book["language"] if is_edit else "", placeholder="English")

    with col2:
        book_format = st.text_input("Format", book["format"] if is_edit else "", placeholder="Pdf / Kindle / Physical")
        source = st.text_input("Source", book["source"] if is_edit else "", placeholder="Shelf / Google Drive")
        pages = st.number_input("Pages", min_value=1, step=1, value=int(book["pages"]) if is_edit else 1)
        rating = st.slider("Rating", 0.0, 5.0, float(book["rating"]) if is_edit else 0.0, 0.05)

    st.subheader("📌 Reading Context")

    col3, col4 = st.columns(2)

    with col3:
        priority = st.selectbox(
            "Priority",
            ["High", "Medium", "Low"],
            index=["High", "Medium", "Low"].index(book["priority"]) if is_edit else 0
        )

    with col4:
        status = st.selectbox(
            "Status",
            ["To-Read", "Reading", "Read", "Dropped"],
            index=["To-Read", "Reading", "Read", "Dropped"].index(book["status"]) if is_edit else 0
        )

    summary = st.text_area(
        "Summary / Notes",
        book["summary"] if is_edit else "",
        placeholder="What is this book about? Or why do you want to read it?"
    )

    with st.expander("📚 Series (optional)"):
        series_name = st.text_input("Series Name", book["series_name"] if is_edit else "")
        book_number = st.number_input(
            "Book Number",
            min_value=1,
            step=1,
            value=int(book["book_number"]) if is_edit and not pd.isna(book["book_number"]) else 1
        )

    return {
        "title": title,
        "author": author,
        "genre": genre,
        "type": book_type,
        "format": book_format,
        "source": source,
        "status": status,
        "pages": pages,
        "priority": priority,
        "rating": rating,
        "summary": summary,
        "series_name": series_name.strip(),
        "book_number": book_number if series_name.strip() else "",
        "language": language,
        "is_series": "TRUE" if series_name.strip() else "FALSE",
    }

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Overview",
        "📈 Analytics",
        "📖 View Books",
        "➕ Add Book",
        "🔍 Search",
        "✏️ Edit Book",
        "❌ Delete Book",
        "🤖 Recommendations",
        "📌 What Should I Read Next?",
        "🧠 Mood-Based Reading",
    ]
)
# =====================================================
# 🏠 HOME / LANDING PAGE
# =====================================================
if page == "🏠 Home":
    st.markdown(
        """
        # 📚 ShelfSense  
        ### A personal reading decision system
        """
    )

    st.markdown(
        """
        **ShelfSense** is a decision-support application designed to help users choose what to read next 
        based on their personal preferences, reading habits, and current mindset.

        Instead of relying on complex machine learning models, ShelfSense uses 
        **explainable logic and semantic similarity** to provide transparent and meaningful recommendations.
        """
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Key Features")
        st.markdown(
            """
            - 📖 Manage and explore a personal book collection  
            - 📊 Analyze reading patterns and preferences  
            - 🤖 Generate recommendations using filters and similarity scoring  
            - 📌 Decide what to read next based on past enjoyment  
            - 🧠 Receive mood-based reading suggestions  
            """
        )

    with col2:
        st.subheader("🧭 How to Use ShelfSense")
        st.markdown(
            """
            1. Add books to your library using **➕ Add Book**  
            2. Review insights in **📊 Overview** and **📈 Analytics**  
            3. Use **🤖 Recommendations** to shortlist options  
            4. Try **📌 What Should I Read Next?** for guided decisions  
            5. Use **🧠 Mood-Based Reading** when unsure what to pick  
            """
        )

    st.markdown("---")

    st.caption(
        "ShelfSense focuses on explainable decision logic rather than heavy machine learning, "
        "making recommendations easy to understand and adapt."
    )

    with st.expander("ℹ️ About this project"):
        st.markdown(
            """
            **ShelfSense** was developed as a personal project to explore:
            - data-driven decision making  
            - user-centered interface design  
            - content-based recommendation logic  

            **Technologies used:**
            - Python  
            - Streamlit  
            - Pandas  
            - Scikit-learn (TF-IDF & cosine similarity)

            This project demonstrates how structured data and simple algorithms can be combined 
            to build practical, human-centered applications.
            """
        )


# =====================================================
# 📊 OVERVIEW
# =====================================================
if page == "📊 Overview":
    df = load_books()
    st.title("📊 Library Overview")
    st.caption("A snapshot of your reading habits")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Total Books", len(df))
    c2.metric("⭐ Avg Rating", round(df["rating"].mean(), 2))
    c3.metric("📄 Avg Pages", int(df["pages"].mean()))
    c4.metric("⏱ Avg Reading Time", round(df["reading_time_hrs"].mean(), 2))

    st.markdown("---")
    st.subheader("📖 Reading Progress")

    fig, ax = plt.subplots()
    df["status"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# =====================================================
# 📈 ANALYTICS
# =====================================================
elif page == "📈 Analytics":
    df = load_books()
    st.title("📈 Reading Analytics")
    st.caption("Patterns and tendencies in your reading choices")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📚 Status")
        st.bar_chart(df["status"].value_counts())

        st.subheader("🎯 Priority")
        st.bar_chart(df["priority"].value_counts())

    with col2:
        st.subheader("⭐ Rating Quality")
        st.bar_chart(df["rating_category"].value_counts())

        st.subheader("📄 Book Length")
        st.bar_chart(df["pages_category"].value_counts())

# =====================================================
# 📖 VIEW BOOKS
# =====================================================
elif page == "📖 View Books":
    st.title("📖 Your Library")
    st.caption("All books in your collection")
    st.dataframe(load_books(), use_container_width=True)

# =====================================================
# ➕ ADD BOOK
# =====================================================
elif page == "➕ Add Book":
    st.title("➕ Add a New Book")
    st.caption("Add a book with minimal effort — fill what you know")

    df = load_books()

    with st.form("add_book"):
        data = book_form()
        submit = st.form_submit_button("Add Book")

        if submit:
            reading_time = round(data["pages"] / 30, 2)
            data.update({
                "rating_category": rating_category(data["rating"]),
                "pages_category": pages_category(data["pages"]),
                "reading_time_hrs": reading_time,
                "recommendation_score": round(data["rating"] + priority_weight(data["priority"]), 2),
                "predicted_score": round(
                    data["rating"] + priority_weight(data["priority"]) +
                    (0.3 if data["is_series"] == "TRUE" else 0), 2
                ),
            })

            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            save_books(df)
            st.success("✅ Book added successfully")
            st.rerun()

# =====================================================
# 🔍 SEARCH
# =====================================================
elif page == "🔍 Search":
    st.title("🔍 Search Books")
    st.caption("Search by title, author, or genre")

    df = load_books()
    c1, c2, c3 = st.columns(3)

    t = c1.text_input("Title")
    a = c2.text_input("Author")
    g = c3.text_input("Genre")

    res = df.copy()
    if t:
        res = res[res["title"].str.contains(t, case=False, na=False)]
    if a:
        res = res[res["author"].str.contains(a, case=False, na=False)]
    if g:
        res = res[
            res["genre"].str.lower().str.split(",")
            .apply(lambda x: g.lower() in [i.strip() for i in x])
        ]

    st.markdown(f"### 🔎 Results ({len(res)})")
    st.dataframe(res, use_container_width=True)

# =====================================================
# ✏️ EDIT BOOK
# =====================================================
elif page == "✏️ Edit Book":
    df = load_books()
    st.title("✏️ Edit Book")
    st.caption("Update details as your opinion changes")

    title = st.selectbox("Select a book", df["title"])
    idx = df[df["title"] == title].index[0]

    with st.form("edit_book"):
        data = book_form(df.loc[idx])
        submit = st.form_submit_button("Update Book")

        if submit:
            data.update({
                "rating_category": rating_category(data["rating"]),
                "pages_category": pages_category(data["pages"]),
                "reading_time_hrs": round(data["pages"] / 30, 2),
                "recommendation_score": round(data["rating"] + priority_weight(data["priority"]), 2),
                "predicted_score": round(
                    data["rating"] + priority_weight(data["priority"]) +
                    (0.3 if data["is_series"] == "TRUE" else 0), 2
                ),
            })

            for k, v in data.items():
                df.loc[idx, k] = v

            save_books(df)
            st.success("✅ Book updated")
            st.rerun()

# =====================================================
# ❌ DELETE BOOK
# =====================================================
elif page == "❌ Delete Book":
    df = load_books()
    st.title("❌ Delete Book")
    st.caption("Remove a book permanently")

    title = st.selectbox("Select a book to delete", df["title"])
    idx = df[df["title"] == title].index[0]

    if st.checkbox("I understand this cannot be undone"):
        if st.button("Delete Book"):
            df = df.drop(idx).reset_index(drop=True)
            save_books(df)
            st.success("✅ Book deleted")
            st.rerun()

# =====================================================
# 🤖 RECOMMENDATIONS
# =====================================================
elif page == "🤖 Recommendations":
    st.title("🤖 Recommendations")
    st.caption("Shortlist books using explicit filters")

    df = load_books()

    with st.expander("🎯 Filters"):
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
        priorities = st.multiselect("Priority", df["priority"].unique(), df["priority"].unique())
        types = st.multiselect("Type", df["type"].unique(), df["type"].unique())
        formats = st.multiselect("Format", df["format"].unique(), df["format"].unique())

    filtered = df[
        (df["rating"] >= min_rating) &
        (df["priority"].isin(priorities)) &
        (df["type"].isin(types)) &
        (df["format"].isin(formats))
    ].sort_values("predicted_score", ascending=False)

    for _, row in filtered.head(10).iterrows():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {row['title']}")
            st.caption(f"{row['author']} • {row['genre']}")
        with col2:
            st.metric("⭐ Rating", row["rating"])
            st.metric("⏱ Hrs", row["reading_time_hrs"])
        st.markdown("---")

# =====================================================
# 📌 WHAT SHOULD I READ NEXT? (FIXED)
# =====================================================
elif page == "📌 What Should I Read Next?":
    df = load_books()
    st.title("📌 What Should I Read Next?")
    st.caption("A clear next-step recommendation based on your recent taste")

    # -------------------------------
    # User input
    # -------------------------------
    last_liked = st.selectbox(
        "Which book did you enjoy recently?",
        df["title"]
    )

    reference = df[df["title"] == last_liked].iloc[0]

    # -------------------------------
    # Candidate pool
    # -------------------------------
    candidates = df[
        (df["status"].isin(["To-Read", "Reading"])) &
        (df["title"] != last_liked)
    ].copy()

    if candidates.empty:
        st.warning("No unread books available to recommend.")
        st.stop()

    # -------------------------------
    # Text similarity (CORRECT)
    # -------------------------------
    reference_text = (
        str(reference["summary"]) + " " +
        str(reference["genre"]) + " " +
        str(reference["author"])
    )

    candidate_texts = (
        candidates["summary"].fillna("") + " " +
        candidates["genre"].fillna("") + " " +
        candidates["author"].fillna("")
    )

    corpus = [reference_text] + candidate_texts.tolist()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    candidates["similarity"] = similarities

    # -------------------------------
    # Genre overlap (organization)
    # -------------------------------
    liked_genres = set(
        g.strip().lower()
        for g in str(reference["genre"]).split(",")
    )

    def genre_overlap(candidate_genre):
        cg = set(g.strip().lower() for g in str(candidate_genre).split(","))
        return len(liked_genres.intersection(cg))

    candidates["genre_overlap"] = candidates["genre"].apply(genre_overlap)

    # -------------------------------
    # Final scoring
    # -------------------------------
    candidates["final_score"] = (
        candidates["predicted_score"] +
        candidates["similarity"] +
        (candidates["genre_overlap"] * 0.4) +
        candidates["status"].apply(lambda x: 0.3 if x == "Reading" else 0) +
        candidates["is_series"].apply(lambda x: 0.5 if x == "TRUE" else 0)
    )

    candidates = candidates.sort_values("final_score", ascending=False)

    # -------------------------------
    # 🥇 Best recommendation
    # -------------------------------
    best = candidates.iloc[0]

    st.markdown("---")
    st.subheader("✨ Best Next Read")

    st.markdown(
        f"""
        ### **{best['title']}**
        👤 {best['author']}  
        ⭐ Rating: {best['rating']}  
        🎯 Priority: {best['priority']}  
        📚 Genre: {best['genre']}

        **Why this book?**
        - Similar themes to *{last_liked}*
        - High predicted enjoyment
        - {'Already in progress' if best['status'] == 'Reading' else 'Easy to start'}
        """
    )

    # -------------------------------
    # 📚 Similar genre options
    # -------------------------------
    similar_genre = candidates[
        candidates["genre_overlap"] > 0
    ].iloc[1:4]

    if not similar_genre.empty:
        st.subheader("📚 Similar Genre Picks")
        for _, row in similar_genre.iterrows():
            st.markdown(
                f"- **{row['title']}** ({row['genre']})"
            )

    # -------------------------------
    # 🌱 Exploration options
    # -------------------------------
    exploration = candidates[
        candidates["genre_overlap"] == 0
    ].iloc[:3]

    if not exploration.empty:
        st.subheader("🌱 Try Something Different")
        for _, row in exploration.iterrows():
            st.markdown(
                f"- **{row['title']}** ({row['genre']})"
            )

# =====================================================
# 🧠 MOOD-BASED READING
# =====================================================
elif page == "🧠 Mood-Based Reading":
    st.title("🧠 Mood-Based Reading")
    st.caption("Adjust how you feel — the system adapts")

    df = load_books()
    candidates = df[df["status"].isin(["To-Read", "Reading"])].copy()
    candidates["mood_score"] = 0.0

    st.subheader("How are you feeling right now?")
    light = st.slider("🌱 Light & Easy", 0, 5, 0)
    comfort = st.slider("🛋 Comfort", 0, 5, 0)
    fast = st.slider("⚡ Fast-Paced", 0, 5, 0)
    thought = st.slider("🧠 Thoughtful", 0, 5, 0)
    intense = st.slider("🔥 Intense", 0, 5, 0)

    if light:
        candidates.loc[candidates["pages_category"] == "Short", "mood_score"] += light * 0.4
    if comfort:
        candidates.loc[candidates["is_series"] == "TRUE", "mood_score"] += comfort * 0.6
    if fast:
        candidates.loc[candidates["genre"].str.contains("thriller|mystery", case=False, na=False), "mood_score"] += fast * 0.6
    if thought:
        candidates.loc[candidates["genre"].str.contains("philosophy|psychology", case=False, na=False), "mood_score"] += thought * 0.6
    if intense:
        candidates.loc[candidates["pages_category"].isin(["Long", "Very Long"]), "mood_score"] += intense * 0.7

    candidates["final_score"] = candidates["predicted_score"] + candidates["mood_score"]

    for _, row in candidates.sort_values("final_score", ascending=False).head(5).iterrows():
        st.markdown(f"### {row['title']}")
        st.caption(f"{row['author']} • Mood Score: {row['mood_score']:.2f}")