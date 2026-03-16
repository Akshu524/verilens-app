import os
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup


NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()

TRUSTED_DOMAINS = [
    "indianexpress.com",
    "bbc.com",
    "reuters.com",
    "thehindu.com",
    "ndtv.com",
    "hindustantimes.com",
    "cnn.com",
    "nytimes.com",
    "aljazeera.com",
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


st.set_page_config(
    page_title="VeriLens",
    page_icon="VL",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg-1: #07111a;
        --bg-2: #0d2430;
        --bg-3: #153847;
        --card: rgba(255, 255, 255, 0.07);
        --border: rgba(255, 255, 255, 0.10);
        --accent-1: #7df9ff;
        --accent-2: #7effa1;
        --text: #f4fbff;
        --muted: #b9d3df;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(125, 249, 255, 0.14), transparent 35%),
            radial-gradient(circle at bottom right, rgba(126, 255, 161, 0.12), transparent 30%),
            linear-gradient(135deg, var(--bg-1), var(--bg-2), var(--bg-3));
        color: var(--text);
    }

    h1, h2, h3 {
        color: var(--text) !important;
    }

    .hero-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        text-align: center;
        color: var(--muted);
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }

    .panel {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        backdrop-filter: blur(8px);
    }

    .stButton > button {
        background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
        color: #081018;
        border: none;
        border-radius: 999px;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
    }

    .stTextInput > div > div > input,
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.06);
        color: var(--text);
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='hero-title'>VeriLens</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-subtitle'>AI-assisted misinformation verification with URL extraction, cross-source checks, and credibility signals.</div>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    """Loads the emotion model lazily so import failures do not crash the app."""
    try:
        from transformers import pipeline

        return pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
    except Exception:
        return None


def detect_emotion_fallback(text: str) -> Tuple[str, float]:
    text_lower = text.lower()

    fear_words = ["panic", "danger", "fear", "terrified", "scared", "threat", "warning"]
    anger_words = ["outrage", "angry", "rage", "corrupt", "fraud", "shocking"]
    joy_words = ["happy", "celebrate", "win", "success", "hope", "joy"]
    sadness_words = ["sad", "tragic", "death", "grief", "loss", "cry"]

    scores = {
        "fear": sum(word in text_lower for word in fear_words),
        "anger": sum(word in text_lower for word in anger_words),
        "joy": sum(word in text_lower for word in joy_words),
        "sadness": sum(word in text_lower for word in sadness_words),
    }

    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]
    if top_score == 0:
        return "neutral", 0.50
    return top_label, min(0.95, 0.55 + top_score * 0.08)


def detect_emotion(text: str) -> Tuple[str, float, str]:
    model = load_emotion_model()
    if model is None:
        label, score = detect_emotion_fallback(text)
        return label, score, "heuristic"

    try:
        output = model(text[:2000])
        scores = output[0] if output and isinstance(output[0], list) else output
        top_emotion = max(scores, key=lambda item: item["score"])
        return top_emotion["label"], float(top_emotion["score"]), "model"
    except Exception:
        label, score = detect_emotion_fallback(text)
        return label, score, "heuristic"


def normalize_domain(url: str) -> str:
    domain = urlparse(url).netloc.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def is_valid_http_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def extract_article_content(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Returns title, extracted text, error message."""
    if not is_valid_http_url(url):
        return None, None, "Please enter a valid URL starting with http:// or https://"

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=12)
        response.raise_for_status()
    except requests.RequestException as exc:
        return None, None, f"Could not fetch the article: {exc}"

    try:
        soup = BeautifulSoup(response.text, "lxml")
    except Exception:
        soup = BeautifulSoup(response.text, "html.parser")

    title = ""
    if soup.title and soup.title.text:
        title = soup.title.text.strip()

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
        if p.get_text(" ", strip=True)
    ]
    article_text = " ".join(paragraphs)

    if len(article_text) < 200:
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            article_text = f"{title}. {meta_desc['content']}".strip()

    if len(article_text) < 80:
        return title or "Untitled Article", None, "Could not extract sufficient article content."

    return title or "Untitled Article", article_text, None


def verify_with_newsapi(query_text: str) -> Tuple[List[Dict[str, str]], str]:
    if not NEWS_API_KEY:
        return [], "NewsAPI key not configured. Add NEWS_API_KEY in Streamlit secrets or environment variables."

    short_query = " ".join(query_text.split()[:8]).strip()
    if not short_query:
        return [], "No query text available for live verification."

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": short_query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=8)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return [], f"Live verification request failed: {exc}"
    except ValueError:
        return [], "NewsAPI returned an unreadable response."

    if data.get("status") != "ok":
        return [], data.get("message", "NewsAPI did not return a valid result.")

    articles = data.get("articles", [])
    results = []
    for article in articles[:3]:
        results.append(
            {
                "title": article.get("title", "Untitled"),
                "source": article.get("source", {}).get("name", "Unknown source"),
                "url": article.get("url", ""),
            }
        )

    if not results:
        return [], "No matching live coverage found."

    return results, "Live verification succeeded."


def compute_credibility(
    is_trusted_source: bool,
    live_matches: int,
    emotion_label: str,
    emotion_score: float,
) -> float:
    score = 0.30

    if is_trusted_source:
        score += 0.35

    score += min(live_matches, 3) * 0.16

    if emotion_label in {"anger", "fear"} and emotion_score >= 0.80:
        score -= 0.15
    elif emotion_label in {"joy", "neutral"}:
        score += 0.05

    return max(0.0, min(1.0, score))


def render_verification_result(score: float):
    if score >= 0.75:
        st.success("High credibility signal")
    elif score >= 0.50:
        st.warning("Moderate credibility signal")
    else:
        st.error("Low credibility signal")
        st.warning(
            "Please do not forward this content until it is verified with official reporting or multiple trusted sources."
        )


st.markdown("---")

left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.header("URL Analysis")
    url_input = st.text_input("Paste a news article URL")

    if st.button("Analyze URL", use_container_width=True):
        if not url_input.strip():
            st.warning("Please enter a URL to analyze.")
        else:
            with st.spinner("Extracting article and checking coverage..."):
                article_title, article_text, extraction_error = extract_article_content(url_input.strip())

            if extraction_error:
                st.error(extraction_error)
            else:
                domain = normalize_domain(url_input)
                trusted_source = any(
                    domain == trusted or domain.endswith(f".{trusted}") for trusted in TRUSTED_DOMAINS
                )

                emotion_label, emotion_score, emotion_source = detect_emotion(article_text)
                live_results, live_status = verify_with_newsapi(article_title or article_text[:120])
                credibility_score = compute_credibility(
                    is_trusted_source=trusted_source,
                    live_matches=len(live_results),
                    emotion_label=emotion_label,
                    emotion_score=emotion_score,
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Emotion", emotion_label.title())
                m2.metric("Credibility", f"{credibility_score * 100:.0f}%")
                m3.metric("Source Trust", "Trusted" if trusted_source else "Unverified")

                render_verification_result(credibility_score)

                st.caption(f"Emotion detection mode: {emotion_source}")
                st.markdown(f"**Article title:** {article_title}")
                st.markdown(f"**Domain:** {domain}")

                st.markdown("#### Live cross-source coverage")
                if live_results:
                    for item in live_results:
                        source = item["source"]
                        title = item["title"]
                        link = item["url"]
                        if link:
                            st.markdown(f"- [{title}]({link}) ({source})")
                        else:
                            st.markdown(f"- {title} ({source})")
                else:
                    st.info(live_status)

                with st.expander("Extracted article preview"):
                    st.write(article_text[:2000] + ("..." if len(article_text) > 2000 else ""))
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.header("Text Analysis")
    user_text = st.text_area("Paste a news claim, article excerpt, or forwarded message", height=260)

    if st.button("Analyze Text", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            main_claim = user_text.split(".")[0].strip() or user_text[:120]
            emotion_label, emotion_score, emotion_source = detect_emotion(user_text)
            live_results, live_status = verify_with_newsapi(main_claim)
            credibility_score = compute_credibility(
                is_trusted_source=False,
                live_matches=len(live_results),
                emotion_label=emotion_label,
                emotion_score=emotion_score,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Emotion", emotion_label.title())
            m2.metric("Emotion Strength", f"{emotion_score * 100:.0f}%")
            m3.metric("Credibility", f"{credibility_score * 100:.0f}%")

            render_verification_result(credibility_score)

            st.caption(f"Emotion detection mode: {emotion_source}")

            if emotion_label in {"anger", "fear"} and emotion_score >= 0.80:
                st.info(
                    "This text uses strong emotional language. That does not prove it is false, but it does mean it deserves extra verification."
                )

            st.markdown("#### Live cross-source coverage")
            if live_results:
                for item in live_results:
                    source = item["source"]
                    title = item["title"]
                    link = item["url"]
                    if link:
                        st.markdown(f"- [{title}]({link}) ({source})")
                    else:
                        st.markdown(f"- {title} ({source})")
            else:
                st.info(live_status)
    st.markdown("</div>", unsafe_allow_html=True)


with st.expander("Deployment notes"):
    st.write("Set `NEWS_API_KEY` in Streamlit Cloud secrets if you want live headline verification.")
    st.write("The app still runs without it, but cross-source checks will be limited.")
