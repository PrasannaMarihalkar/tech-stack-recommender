# Tech Stack Recommender 🚀

**DecodeLabs Industrial Training — AI Project 3**  
A content-based filtering recommendation engine that maps user skills to career paths using TF-IDF vectorization and Cosine Similarity.

## How It Works

The system implements a 4-step ranking pipeline:

1. **Ingestion** — Captures user skills (minimum 3)
2. **Scoring** — Computes TF-IDF weighted Cosine Similarity against 20 job role profiles
3. **Sorting** — Ranks all job roles by similarity score (descending)
4. **Filtering** — Returns the Top-N most relevant career paths

## Tech Stack

- Python 3.10+
- scikit-learn (TfidfVectorizer, cosine_similarity)
- pandas / numpy

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/tech-stack-recommender.git
cd tech-stack-recommender
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Interactive mode:**
```bash
python main.py
```

**CLI mode:**
```bash
python main.py --skills Python Docker AWS Kubernetes --top 3
```

## Sample Output

```
  🥇 RANK 1
  Role        : DevOps Engineer
  Match Score : 87.3%
  Key Matches : aws, docker, kubernetes

  🥈 RANK 2
  Role        : Cloud Architect
  Match Score : 74.1%
  Key Matches : aws, kubernetes

  🥉 RANK 3
  Role        : Site Reliability Engineer
  Match Score : 61.8%
  Key Matches : aws, docker, kubernetes
```

## Run Tests

```bash
python -m pytest tests/ -v
```