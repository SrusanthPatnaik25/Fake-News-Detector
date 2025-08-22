# 📰 Fake News Fact Checker (Semantic Match)

**Check whether a news headline or claim is REAL, PARTIALLY TRUE, or FAKE by matching it against live headlines from trusted news sources.**

---

## 🚀 About

This app uses **semantic similarity** with embeddings from the [Sentence Transformers](https://www.sbert.net/) library to compare your input claim with **live headlines** fetched from popular news sources including:

- BBC  
- Times of India  
- Reuters  
- NDTV  
- The Hindu  
- Indian Express  
- New York Times  
- Hindustan Times  
- Moneycontrol  
- ANI News  
- Economic Times  
- The Guardian  
- Al Jazeera  
- CNN  
- Times Now  

It then predicts:

- **REAL** → Best match similarity ≥ 0.80  
- **PARTIALLY TRUE** → Similarity between 0.60 and 0.80  
- **FAKE / UNVERIFIED** → Similarity < 0.60  

---

## 📌 Features

- Fetches **live news headlines** via RSS feeds  
- Builds **semantic embeddings** for all headlines  
- Finds **closest matches** to the input claim  
- Shows **top K matches** with similarity scores  
- Displays **source and URL** of matching news  
- Dynamic **Gradio UI** for entering claims and analyzing in real time  

---

## 🖥 Usage

1. Select the news sources you want to use for checking.  
2. Click **Refresh Headlines Index** to fetch and index the latest headlines.  
3. Enter your news claim or headline in the textbox.  
4. Click **Analyze Claim** to see:

   - **Prediction** (REAL, PARTIALLY TRUE, or FAKE / UNVERIFIED)  
   - **Best matching headline** with source and similarity score  
   - **Top K matches** in a table with clickable URLs  

---

## ⚡ Quick Start (Locally)

```bash
# Clone the repository
git clone https://huggingface.co/spaces/<your-username>/<your-space-name>

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
