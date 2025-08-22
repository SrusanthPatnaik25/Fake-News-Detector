# Fake News Fact Checker (Semantic Match)

An AI-powered fact-checking web application that uses semantic similarity to detect fake news by comparing user-submitted claims or headlines against live news headlines from trusted sources.

The app fetches **live headlines** from multiple trusted news sources such as BBC, Times of India, Reuters, NDTV, The Hindu, Indian Express, New York Times, and more, builds **semantic embeddings** using Sentence Transformers, and finds the **closest matches** to the user’s input claim.

## Features

- Real-time fetching of news headlines using RSS feeds from multiple trusted sources.
- Semantic similarity matching using pre-trained Sentence Transformers (all-MiniLM-L6-v2).
- Categorizes claims based on cosine similarity into:
  - **REAL** (similarity ≥ 0.80)
  - **PARTIALLY TRUE** (similarity 0.60–0.80)
  - **FAKE / UNVERIFIED** (similarity < 0.60)
- User-friendly interface powered by Gradio for easy interaction.
- Displays best-matching headlines with source, similarity score, and URL link.
- Shows top-K closest matching headlines in a tabular format.
