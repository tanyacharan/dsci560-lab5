# Reddit Post Clustering System

A machine learning pipeline for clustering Reddit posts using TF-IDF vectorization, Singular Value Decomposition (SVD), and KMeans clustering. The system provides real-time classification capabilities and comprehensive text processing optimized for Reddit content.

## Features

### Core Functionality
- **Text Processing**: Reddit-specific preprocessing including URL, mention, and subreddit handling
- **Feature Extraction**: TF-IDF vectorization with configurable n-grams
- **Dimensionality Reduction**: SVD for efficient embedding representation
- **Clustering**: KMeans clustering with configurable cluster count
- **Real-time Processing**: Continuous monitoring and classification of new posts
- **Visualization**: UMAP-based 2D cluster visualization

### Text Processing Pipeline
- HTML tag and entity removal
- Reddit-specific token replacement (URLs → URL, /u/user → USER, /r/subreddit → SUBREDDIT)
- Emoji preservation with spacing
- NLTK-based tokenization and lemmatization
- Stopword filtering with extended common words list
- Preservation of technical abbreviations (GPU, CPU, API, etc.)

## Requirements

### System Dependencies
- Python 3.8+
- MySQL 5.7+

### Python Dependencies
```
mysql-connector-python
numpy
pandas
scikit-learn
nltk
python-dotenv
joblib
umap-learn
matplotlib
```

### NLTK Data Requirements
```python
# Download required NLTK data
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/tanyacharan/dsci560-lab4.git
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the project root:
```env
DB_HOST=localhost
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=reddit
```

Note: Currently the DB details are hard-coded - this will be reflected.

4. **Initialize NLTK data**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

5. **Prepare the database**
Ensure your MySQL database has a `posts` table with the required structure (see Database Schema section).

## Database Schema

### Required Table Structure
```sql
CREATE TABLE posts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    title TEXT,
    selftext TEXT,
    ocr_text TEXT,
    embedding MEDIUMTEXT,
    cluster_id INT,
    INDEX idx_cluster_id (cluster_id)
);
```

The system will automatically add missing columns and indexes if they don't exist.

## Usage

### Initial Training
Train the TF-IDF and SVD models, then perform clustering:
```bash
# Train models and generate embeddings
python clust_improved.py --train

# Run KMeans with 10 clusters
python clust_improved.py --kmeans 10
```

### Update Operations
Process new posts that don't have embeddings:
```bash
python clust_improved.py --update-emb
```

### Real-time Processing
Run continuous monitoring with 5-minute intervals:
```bash
python clust_improved.py --realtime 5
```

### Single Message Classification
Classify an individual message:
```bash
python clust_improved.py --classify "What's the best Python IDE for data science?"
```

### Cluster Analysis
Generate keyword analysis for each cluster:
```bash
python clust_improved.py --keywords
```

### Combined Operations
Multiple operations can be combined in a single command:
```bash
python clust_improved.py --train --kmeans 8 --keywords
```

## Command-Line Options

| Option | Argument | Description |
|--------|----------|-------------|
| `--train` | - | Train TF-IDF and SVD models, generate embeddings |
| `--update-emb` | - | Generate embeddings for posts without embeddings |
| `--kmeans` | K | Run KMeans clustering with K clusters |
| `--realtime` | MINUTES | Run real-time processing loop with specified interval |
| `--classify` | TEXT | Classify a message into existing clusters |
| `--keywords` | - | Generate cluster keywords analysis |
| `--verbose`, `-v` | - | Enable verbose logging |

## Architecture

### Component Overview

1. **Config Class**
   - Centralized configuration management
   - Environment variable loading
   - Model path definitions
   - ML hyperparameter storage

2. **TextProcessor**
   - Reddit-specific text cleaning
   - NLTK-based tokenization and lemmatization
   - Regex fallback for environments without NLTK
   - Stopword filtering with domain-specific extensions

3. **DatabaseManager**
   - Connection pooling
   - Schema verification and migration
   - Batch processing support
   - Error handling and recovery

4. **EmbeddingManager**
   - TF-IDF vectorization (max 50,000 features)
   - SVD dimensionality reduction (256 components)
   - Model persistence and loading
   - Batch embedding generation

5. **ClusteringManager**
   - KMeans clustering implementation
   - UMAP visualization generation
   - Cluster keyword extraction
   - Database cluster ID updates

6. **RealtimeProcessor**
   - Continuous monitoring loop
   - New post classification
   - Similarity search within clusters
   - Single message classification

### Processing Pipeline

```
Raw Reddit Post
    ↓
Text Extraction (title + selftext + ocr_text)
    ↓
Reddit-Specific Preprocessing
    ↓
Tokenization & Lemmatization
    ↓
TF-IDF Vectorization (50,000 features)
    ↓
SVD Reduction (256 dimensions)
    ↓
KMeans Clustering
    ↓
Database Storage
```

## Configuration

### Model Parameters (in Config class)
- `MAX_FEATURES`: 50,000 - Maximum vocabulary size for TF-IDF
- `N_COMPONENTS`: 256 - SVD embedding dimensions
- `MIN_DF`: 3 - Minimum document frequency for terms
- `NGRAM_RANGE`: (1, 2) - Unigrams and bigrams
- `BATCH_SIZE`: 500 - Database write batch size
- `DEFAULT_K_CLUSTERS`: 8 - Default number of clusters

### File Structure
```
.
├── clust_improved.py      # Main application code
├── models/               # Trained model storage
│   ├── tfidf_vectorizer.pkl
│   ├── svd.pkl
│   └── kmeans.pkl
├── cluster_analysis.csv   # Generated cluster keywords
└── clusters_visualization.png  # UMAP visualization
```

## Output Files

1. **models/** - Serialized ML models
2. **cluster_analysis.csv** - Top keywords per cluster
3. **clusters_visualization.png** - 2D UMAP projection of clusters

## Error Handling

The system includes comprehensive error handling for:
- Database connection failures
- Malformed JSON embeddings
- Missing NLTK dependencies (falls back to regex)
- Model loading failures
- Batch processing interruptions

## Logging

Logging is configured with timestamp, module name, and severity level. Use `--verbose` flag for DEBUG level output.

## Troubleshooting

### Common Issues

1. **"Models not found" error**
   - Run `--train` first to generate initial models

2. **Database connection failures**
   - Verify `.env` file configuration
   - Check MySQL service status
   - Ensure database and table exist

3. **Memory errors with large datasets**
   - Reduce `MAX_FEATURES` in Config class
   - Process posts in smaller batches
   - Increase system swap space

4. **NLTK not available warning**
   - Install NLTK: `pip install nltk`
   - Download data: `python -c "import nltk; nltk.download('stopwords')"`
