# 📰 RockyBot: News Research Tool

RockyBot is an AI-powered News Research Tool that allows users to load news articles from URLs, generate summaries, and ask questions about the article content using **Google Gemini**, **LangChain**, and **FAISS**.

## 🚀 Live Demo

🔗 https://news-research-tool-bjpadm82yb26sw4bmbrvws.streamlit.app/

---

## Features

- Load news articles from URLs.
- Extract article content using LangChain's URL Loader.
- Generate vector embeddings using **Google Gemini Embeddings**.
- Store embeddings efficiently with **FAISS**.
- Ask questions about uploaded news articles.
- Receive AI-generated answers with relevant source information.

---

## Technologies Used

- Python
- Streamlit
- LangChain
- Google Gemini API
- FAISS
- Python Dotenv

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the project folder

```bash
cd YOUR_REPOSITORY
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file (for local development)

```env
GOOGLE_API_KEY=your_google_api_key_here
```

---

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL displayed in your terminal.

---

## How It Works

1. Enter one or more news article URLs.
2. Click **Process URLs**.
3. The application:
   - Extracts article content.
   - Splits the text into chunks.
   - Generates embeddings using Google Gemini.
   - Stores embeddings in FAISS.
4. Ask questions about the uploaded articles.
5. Receive AI-generated answers based on the article content.

---

## Project Structure

```
News-Research-Tool/
│
├── app.py
├── requirements.txt
├── .env
├── faiss_store.pkl
├── rockybot.jpg
└── README.md
```

---

## Future Improvements

- Support PDF and DOCX files.
- Multi-document chat.
- Conversation memory.
- Chat history.
- Source highlighting.

---

## Author

**Sakshi Raut**

B.Tech CSE (AI & ML)

Vishwakarma University

---

## License

This project is developed for educational purposes.
