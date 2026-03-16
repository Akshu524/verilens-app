# VeriLens

VeriLens is an AI-powered misinformation verification web app built with Streamlit. It helps users analyze news article URLs and text-based claims by checking source trust, extracting article content, detecting emotional tone, and verifying whether similar coverage appears across live news sources.

## Features

- URL-based news article analysis
- Text claim verification
- Emotional tone detection
- Trusted source identification
- Live cross-source news verification using NewsAPI
- Credibility scoring based on source trust, emotional intensity, and matching news coverage
- Clean and responsive Streamlit interface

## How It Works

### URL Analysis
Users can paste a news article link, and VeriLens will:
- Extract the article title and text
- Detect the emotional tone of the content
- Check whether the domain belongs to a trusted source
- Search for similar live news coverage
- Estimate a credibility score

### Text Analysis
Users can paste a forwarded message, claim, or article excerpt, and VeriLens will:
- Detect emotional tone
- Search for matching live news coverage
- Generate a credibility score
- Warn users when the content has low credibility signals

## Tech Stack

- Python
- Streamlit
- Requests
- BeautifulSoup
- Transformers pipeline with DistilRoBERTa emotion model
- NewsAPI

## Project Structure

```bash
verilens-app/
│-- app.py
│-- requirements.txt
│-- README.md
