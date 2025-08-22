import gradio as gr
import feedparser
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# ------------------------------------
# Trusted sources (RSS)
# ------------------------------------
RSS_FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "TOI": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
    "Reuters": "http://feeds.reuters.com/reuters/topNews",
    "NDTV": "https://feeds.feedburner.com/ndtvnews-top-stories",
    "The Hindu": "https://www.thehindu.com/feeder/default.rss",
    "Indian Express": "https://indianexpress.com/feed/",
    "New York Times": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "Hindustan Times": "https://www.hindustantimes.com/feeds/rss/topnews/rssfeed.xml",
    "Moneycontrol": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "ANI News": "https://www.aninews.in/rss/feed.aspx?cat=TopStories",
    "Economic Times": "https://economictimes.indiatimes.com/feeds/newsfeeds/1895242001.cms",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "CNN": "http://rss.cnn.com/rss/edition.rss",
    "Times Now": "https://www.timesnownews.com/rss-feed",
}

TOP_K = 5
THRESHOLDS = {
    "REAL": 0.80,     # >= 0.80 -> REAL
    "PARTIAL": 0.60,  # [0.60, 0.80) -> PARTIALLY TRUE
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# ------------------------------------
# Helpers
# ------------------------------------
def fetch_headlines_from_sources(sources):
    """
    Returns list of dicts:
    [{source, title, link, text_for_embed}]
    """
    rows = []
    for src in sources:
        url = RSS_FEEDS.get(src)
        if not url:
            continue
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
        for e in getattr(feed, "entries", []):
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "").strip()
            summary = getattr(e, "summary", "")
            text_for_embed = f"{title} {summary}".strip()
            if title:
                rows.append(
                    {
                        "source": src,
                        "title": title,
                        "link": link,
                        "text_for_embed": text_for_embed,
                    }
                )
    return rows

def build_index(sources):
    """
    Build embeddings over fetched headlines.
    Returns: (rows, embeddings_tensor, message)
    """
    rows = fetch_headlines_from_sources(sources)
    if not rows:
        return rows, None, "No headlines found. Try different sources and refresh again."
    texts = [r["text_for_embed"] for r in rows]
    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    msg = f"Indexed **{len(rows)}** headlines from: {', '.join(sources)}  \nLast refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    return rows, embeddings, msg

def label_from_score(score: float) -> str:
    if score >= THRESHOLDS["REAL"]:
        return "REAL"
    if score >= THRESHOLDS["PARTIAL"]:
        return "PARTIALLY TRUE"
    return "FAKE / UNVERIFIED"

def fact_check_claim(claim, rows, embeddings):
    """
    Returns: (label, best_match_md, topk_dataframe)
    """
    if not claim or embeddings is None or not rows:
        return (
            {"No data": 1.0},
            "No headlines indexed yet. Click **Refresh Headlines Index**.",
            pd.DataFrame(columns=["Similarity", "Source", "Headline", "URL"]),
        )

    # Encode claim, compute similarities
    claim_emb = model.encode(claim, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(claim_emb, embeddings).flatten()

    # Top-K
    k = min(TOP_K, len(rows))
    top_scores, top_idx = sims.topk(k)

    # Build table
    records = []
    for s, i in zip(top_scores.tolist(), top_idx.tolist()):
        r = rows[i]
        records.append(
            {
                "Similarity": round(float(s), 4),
                "Source": r["source"],
                "Headline": r["title"],
                "URL": r["link"],
            }
        )
    df = pd.DataFrame(records)

    # Best match & label
    best_score = float(top_scores[0])
    best_row = rows[int(top_idx[0])]
    label = label_from_score(best_score)
    best_match_md = (
        f"**Best Match:** [{best_row['title']}]({best_row['link']})  \n"
        f"**Source:** {best_row['source']}  \n"
        f"**Similarity:** {best_score:.4f}"
    )

    # Gradio Label expects dict[class]=prob; we show 1.0 for selected class
    return ({label: 1.0}, best_match_md, df)

# ------------------------------------
# UI
# ------------------------------------
with gr.Blocks(title="Fake News Fact Checker (Semantic Match)") as demo:
    gr.Markdown("# 📰 Fake News Fact Checker (Semantic Match)")
    gr.Markdown(
        "Enter a claim/headline. The app fetches **live headlines** from trusted sources "
        "(BBC, Times of India, Reuters, NDTV, The Hindu, Indian Express, New York Times), "
        "builds **semantic embeddings**, and finds the **closest matches**.  \n\n"
        "**Labels** (based on cosine similarity with the best match):  \n"
        "- ≥ **0.80** → **REAL**  \n"
        "- **0.60–0.80** → **PARTIALLY TRUE**  \n"
        "- < **0.60** → **FAKE / UNVERIFIED**"
    )

    with gr.Row():
        sources = gr.CheckboxGroup(
            choices=list(RSS_FEEDS.keys()),
            value=list(RSS_FEEDS.keys()),
            label="News Sources",
        )
        refresh_btn = gr.Button("🔄 Refresh Headlines Index", variant="secondary")

    index_info = gr.Markdown("No index yet. Click **Refresh Headlines Index**.")

    # State to hold current index
    index_rows_state = gr.State([])
    index_emb_state = gr.State(None)

    def do_refresh(src_list):
        rows, emb, msg = build_index(src_list)
        return rows, emb, msg

    refresh_btn.click(
        do_refresh,
        inputs=[sources],
        outputs=[index_rows_state, index_emb_state, index_info],
    )

    gr.Markdown("### 🔎 Check a claim")
    claim = gr.Textbox(
        lines=3,
        placeholder="e.g., Indian government reduces taxes on electronics from November",
        label="Your headline / claim",
    )
    check_btn = gr.Button("Analyze Claim", variant="primary")

    with gr.Row():
        label_out = gr.Label(num_top_classes=3, label="Prediction")
        best_match = gr.Markdown()

    topk_table = gr.Dataframe(
        headers=["Similarity", "Source", "Headline", "URL"],
        label="Top Matches",
        interactive=False,
        wrap=True,
    )

    def run_check(claim_text, rows, emb):
        return fact_check_claim(claim_text, rows, emb)

    check_btn.click(
        run_check,
        inputs=[claim, index_rows_state, index_emb_state],
        outputs=[label_out, best_match, topk_table],
    )

    # Build index automatically on load (with all sources)
    demo.load(
        do_refresh,
        inputs=[sources],
        outputs=[index_rows_state, index_emb_state, index_info],
    )

if __name__ == "__main__":
    demo.launch()
