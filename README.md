# L'ORÉAL CommentSense

> This is the project repository for team **Lamimi** for **L’Oréal x Monash Datathon 2025**. 

## Table of Contents

- [Overview](#overview)
- [Key Modules](#key-modules)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)

---

## Overview

**L'ORÉAL CommentSense** is an end-to-end AI-powered platform for analyzing YouTube comments to extracting actionable marketing insights that drive business value.

## Key Modules

### 1. **YouTube Comment Scraping & Preprocessing**
Extracts and cleans YouTube comments with metadata, removes noise, and prepares data for analysis.

### 2. **Multi-dimensional Comment Analysis Engine**
Performs sentiment analysis, spam detection, relevance scoring, and calculates product resonance scores.

### 3. **Intelligent Product Category Clustering**
Automatically categorizes comments into 5 L'Oréal product segments using semantic clustering.

### 4. **Comment Analysis Dashboard (Power BI)**
Interactive dashboard displaying KPIs, engagement metrics, and category insights in real-time.

### 5. **NL2SQL AI Chatbot (MCP)**
Conversational AI that converts natural language questions to SQL queries, retrieve data from database and provides insights based on queried data.

---

## Tech Stack

### Backend & Processing
- **Python 3.11+**
- **Streamlit**: Web framework for interactive UI
- **Transformers**: Hugging Face models for NLP tasks
- **Sentence Transformers**: Semantic embeddings
- **scikit-learn**: Machine learning backend
- **PyTorch**: Deep learning backend
- **NLTK**: Natural language processing utilities

### Database
- **Supabase**: Managed PostgreSQL backend
- **SQLAlchemy**: ORM for database operations
- **psycopg2**: PostgreSQL adapter

### APIs & External Services
- **Google API Client**: YouTube Data API integration
- **Groq API**: LLM service
- **Power BI**: Interactive dashboards
---

## Repository Structure

```
.                  
├── src/
│   ├── __init__.py
│   ├── backend/                        # Core analysis modules
│   │   ├── compute_actionability.py    # Actionability scoring
│   │   ├── database.py                 # DB connections & queries
│   │   ├── nl2sql_agentic.py          # NL→SQL conversion
│   │   ├── powerbi_utils.py           # Dashboard utilities
│   │   ├── prod_cat_clustering.py     # Product clustering
│   │   ├── relevance_check.py         # Relevance scoring
│   │   ├── report_generator.py        # PDF reports
│   │   ├── sentiment_analysis.py      # Sentiment classification
│   │   ├── spam_detector.py           # Spam detection
│   │   ├── youtube_scrapper.py        # YouTube scraping
│   │   └── __init__.py
│   └── frontend/                       # Streamlit UI
│       ├── main.py                     # App entry point
│       ├── app_pages/                  # Page components
│       │   ├── analyzeComment.py       # Comment analysis page
│       │   ├── chatbot.py              # Chat interface
│       │   ├── dashboard.py            # Insights dashboard
│       │   └── __init__.py
│       ├── assets/                     # Images and static files
│       ├── fonts/                      # Custom fonts
│       ├── styles/                     # CSS styling
│       │   ├── analyze_comment_page.css
│       │   ├── chatbot_page.css
│       │   └── dashboard_page.css
│       └── __init__.py
│
├── README.md                           
├── requirements.txt  
└── .gitignore                          
```

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Supabase project initialized

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Lamimi
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate      # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `src/.env` file with the following variables:
   ```bash
   GROQ_API_KEY=<your-groq-api-key>
   YOUTUBE_API_KEY=<your-youtube-api-key>
   SUPABASE_URL=<your-supabase-url>
   SUPABASE_KEY=<your-supabase-anon-key>
   PG_HOST=<your-postgres-host>
   PG_DATABASE=<your-database-name>
   PG_USER=<your-postgres-user>
   PG_PASSWORD=<your-postgres-password>
   PG_PORT=5432
   ```

5. **Run the application**
   ```bash
   streamlit run src/frontend/main.py
   ```

   The app will be available at `http://localhost:8501`

---
