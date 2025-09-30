import argparse
import json
import os
import time
import re
import logging
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import mysql.connector
from mysql.connector import Error as MySQLError
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- CONFIGURATION ----------
class Config:
    """Centralized configuration management"""
    
    # Database configuration - should be loaded from environment variables
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "user": os.getenv("DB_USER", "phpmyadmin"),
        "password": os.getenv("DB_PASSWORD", ""),  # Passwords not supposed to be hardcoded
        "database": os.getenv("DB_NAME", "reddit"),
        "autocommit": False,
        "raise_on_warnings": True
    }
    
    # Model paths
    MODEL_DIR = Path("models")
    VECT_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
    SVD_PATH = MODEL_DIR / "svd.pkl"
    KMEANS_PATH = MODEL_DIR / "kmeans.pkl"
    
    # ML parameters
    MAX_FEATURES = 50000
    N_COMPONENTS = 256
    MIN_DF = 3
    NGRAM_RANGE = (1, 2)
    RANDOM_STATE = 42
    
    # Processing parameters
    BATCH_SIZE = 500
    DEFAULT_K_CLUSTERS = 8
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories"""
        cls.MODEL_DIR.mkdir(exist_ok=True)

# ---------- TEXT PROCESSING ----------
class TextProcessor:
    """Handles text cleaning and preprocessing for Reddit content"""
    
    # Reddit-specific patterns
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    MENTION_PATTERN = re.compile(r'/u/\w+|u/\w+')
    SUBREDDIT_PATTERN = re.compile(r'/r/\w+|r/\w+')
    HTML_TAG_RE = re.compile(r"<.*?>")
    EMOJI_PATTERN = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE
    )
    WHITESPACE_RE = re.compile(r"\s+")
    
    def __init__(self):
        """Initialize with NLTK components"""
        try:
            from nltk.tokenize import TweetTokenizer
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import stopwords
            
            self.tweet_tokenizer = TweetTokenizer(
                preserve_case=False,
                strip_handles=True,
                reduce_len=True  # Normalize repeated chars (sooooo -> so)
            )
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            # Add additional common words that should be filtered
            self.stop_words.update([
                'would', 'could', 'should', 'might', 'must', 'shall', 'will',
                'like', 'really', 'think', 'know', 'want', 'need', 'say', 'said',
                'get', 'got', 'make', 'made', 'take', 'took', 'come', 'came',
                'go', 'went', 'going', 'thing', 'things', 'way', 'lot', 'use',
                'also', 'one', 'two', 'first', 'last', 'now', 'just', 'even',
                'back', 'still', 'much', 'many', 'well', 'yes', 'yeah'
            ])
            self.use_nltk = True
        except ImportError:
            logger.warning("NLTK not available, falling back to regex tokenization")
            self.use_nltk = False
            self.tweet_tokenizer = None
            self.lemmatizer = None
            self.stop_words = set()
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """Clean text for general processing (backwards compatible)"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = cls.HTML_TAG_RE.sub(" ", text)
        
        # Remove null bytes
        text = text.replace("\x00", " ")
        
        # Normalize whitespace
        text = cls.WHITESPACE_RE.sub(" ", text)
        
        return text.strip()
    
    def preprocess_reddit_text(self, text: str) -> str:
        """Preprocess Reddit-specific content"""
        if not text:
            return ""
        
        # Remove URLs but mark their presence
        text = self.URL_PATTERN.sub(" URL ", text)
        
        # Replace user mentions with generic token
        text = self.MENTION_PATTERN.sub(" USER ", text)
        
        # Replace subreddit mentions with generic token
        text = self.SUBREDDIT_PATTERN.sub(" SUBREDDIT ", text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text, flags=re.IGNORECASE)
        
        # Keep emojis as they may indicate sentiment
        # But add spaces around them
        text = self.EMOJI_PATTERN.sub(r' \g<0> ', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using NLTK if available"""
        if self.use_nltk and self.tweet_tokenizer:
            # Preprocess for Reddit
            text = self.preprocess_reddit_text(text)
            
            # Use TweetTokenizer
            tokens = self.tweet_tokenizer.tokenize(text)
            
            # Filter and lemmatize tokens
            filtered = []
            for token in tokens:
                # Keep Reddit-specific tokens
                if token in ['URL', 'USER', 'SUBREDDIT']:
                    filtered.append(token.lower())
                # Keep common tech abbreviations
                elif token.lower() in ['gpu', 'cpu', 'ram', 'ssd', 'lol', 'tbh', 'imo', 'rtx', 'gtx', 'api', 'ui', 'ux']:
                    filtered.append(token.lower())
                # Process alphabetic tokens
                elif token.isalpha() and len(token) > 2:
                    # Skip stopwords
                    if token.lower() in self.stop_words:
                        continue
                    
                    # Lemmatize the token
                    if self.lemmatizer:
                        lemma = self.lemmatizer.lemmatize(token.lower(), pos='v')  # Try verb
                        lemma = self.lemmatizer.lemmatize(lemma, pos='n')  # Then noun
                        
                        # Only keep if still meaningful after lemmatization
                        if len(lemma) > 2 and lemma not in self.stop_words:
                            filtered.append(lemma)
                    else:
                        filtered.append(token.lower())
            
            return filtered
        else:
            # Fallback to basic regex
            text = self.clean_text(text)
            return re.findall(r'\b[a-z]{3,}\b', text.lower())

# ---------- DATABASE OPERATIONS ----------
class DatabaseManager:
    """Manages database connections and operations"""
    
    @staticmethod
    def get_connection():
        """Get a new database connection"""
        try:
            return mysql.connector.connect(**Config.DB_CONFIG)
        except MySQLError as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    @staticmethod
    def ensure_columns():
        """Ensure required columns exist in the database"""
        column_checks = [
            ("embedding", "ALTER TABLE posts ADD COLUMN embedding MEDIUMTEXT"),
            ("cluster_id", "ALTER TABLE posts ADD COLUMN cluster_id INT")
        ]
        
        index_checks = [
            ("idx_cluster_id", "ALTER TABLE posts ADD INDEX idx_cluster_id (cluster_id)")
        ]
        
        try:
            with DatabaseManager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check and add columns
                    cur.execute("SHOW COLUMNS FROM posts")
                    existing_columns = {row[0] for row in cur.fetchall()}
                    
                    for col_name, query in column_checks:
                        if col_name not in existing_columns:
                            try:
                                cur.execute(query)
                                logger.info(f"Added column: {col_name}")
                            except MySQLError as e:
                                logger.warning(f"Column creation warning for {col_name}: {e}")
                    
                    # Check and add indexes
                    cur.execute("SHOW INDEX FROM posts")
                    existing_indexes = {row[2] for row in cur.fetchall()}  # Key_name is 3rd column
                    
                    for idx_name, query in index_checks:
                        if idx_name not in existing_indexes:
                            try:
                                cur.execute(query)
                                logger.info(f"Added index: {idx_name}")
                            except MySQLError as e:
                                logger.warning(f"Index creation warning for {idx_name}: {e}")
                    
                    conn.commit()
            logger.info("Database schema verified")
        except MySQLError as e:
            logger.error(f"Schema verification failed: {e}")
            raise
    
    @staticmethod
    def fetch_posts_df(limit: Optional[int] = None, only_new: bool = False) -> pd.DataFrame:
        """Fetch posts from database"""
        query = """
            SELECT 
                id, 
                COALESCE(title, '') AS title, 
                COALESCE(selftext, '') AS body, 
                COALESCE(ocr_text, '') AS ocr
            FROM posts
            {where_clause}
            {limit_clause}
        """
        
        where_clause = "WHERE embedding IS NULL" if only_new else ""
        limit_clause = f"LIMIT {int(limit)}" if limit else ""
        query = query.format(where_clause=where_clause, limit_clause=limit_clause)
        
        try:
            with DatabaseManager.get_connection() as conn:
                df = pd.read_sql(query, conn)
                
            # Process text
            df["text"] = df.apply(
                lambda row: TextProcessor.clean_text(
                    f"{row['title']} {row['body']} {row['ocr']}"
                ), 
                axis=1
            )
            
            logger.info(f"Fetched {len(df)} posts from database")
            return df
            
        except MySQLError as e:
            logger.error(f"Failed to fetch posts: {e}")
            raise

# ---------- EMBEDDING OPERATIONS ----------
class EmbeddingManager:
    """Handles embedding generation and storage"""
    
    def __init__(self):
        self.vectorizer = None
        self.svd = None
        self.kmeans = None
        self.text_processor = TextProcessor()
    
    def train_tfidf_svd(self):
        """Train TF-IDF vectorizer and SVD"""
        logger.info("Starting TF-IDF and SVD training...")
        
        df = DatabaseManager.fetch_posts_df()
        if df.empty:
            logger.warning("No posts found in database. Please scrape data first.")
            return
        
        # Train TF-IDF with custom tokenizer
        self.vectorizer = TfidfVectorizer(
            max_features=Config.MAX_FEATURES,
            min_df=Config.MIN_DF,
            ngram_range=Config.NGRAM_RANGE,
            lowercase=True,
            tokenizer=self.text_processor.tokenize,  # Use our hybrid tokenizer
            preprocessor=self.text_processor.preprocess_reddit_text,  # Reddit preprocessing
            token_pattern=None,  # Disable pattern since we use custom tokenizer
            max_df=0.95,  # Ignore terms that appear in >95% of documents
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # Use log(tf + 1) for better scaling
        )
        
        logger.info("Fitting TF-IDF vectorizer...")
        X_tfidf = self.vectorizer.fit_transform(df["text"])
        
        # Train SVD
        self.svd = TruncatedSVD(
            n_components=Config.N_COMPONENTS,
            random_state=Config.RANDOM_STATE,
            algorithm='randomized',
            n_iter=5
        )
        
        logger.info("Fitting SVD...")
        X_embeddings = self.svd.fit_transform(X_tfidf)
        
        # Save models
        self._save_models()
        
        # Write embeddings to database
        self._write_embeddings(df["id"].tolist(), X_embeddings)
        
        logger.info("Training completed successfully")
    
    def update_embeddings_only_new(self):
        """Update embeddings for new posts only"""
        if not self._load_models():
            logger.error("Models not found. Please train first.")
            return
        
        df = DatabaseManager.fetch_posts_df(only_new=True)
        if df.empty:
            logger.info("No new posts to embed")
            return
        
        logger.info(f"Generating embeddings for {len(df)} new posts...")
        X_tfidf = self.vectorizer.transform(df["text"])
        X_embeddings = self.svd.transform(X_tfidf)
        
        self._write_embeddings(df["id"].tolist(), X_embeddings)
    
    def _save_models(self):
        """Save trained models to disk"""
        Config.ensure_dirs()
        
        joblib.dump(self.vectorizer, Config.VECT_PATH)
        joblib.dump(self.svd, Config.SVD_PATH)
        
        logger.info(f"Saved vectorizer to {Config.VECT_PATH}")
        logger.info(f"Saved SVD to {Config.SVD_PATH}")
    
    def _load_models(self) -> bool:
        """Load models from disk"""
        try:
            if not all(path.exists() for path in [Config.VECT_PATH, Config.SVD_PATH]):
                return False
                
            self.vectorizer = joblib.load(Config.VECT_PATH)
            self.svd = joblib.load(Config.SVD_PATH)
            
            if Config.KMEANS_PATH.exists():
                self.kmeans = joblib.load(Config.KMEANS_PATH)
                
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def _write_embeddings(self, ids: List[int], embeddings: np.ndarray):
        """Write embeddings to database"""
        try:
            with DatabaseManager.get_connection() as conn:
                with conn.cursor() as cur:
                    for i, (post_id, embedding) in enumerate(zip(ids, embeddings)):
                        embedding_json = json.dumps(embedding.tolist())
                        cur.execute(
                            "UPDATE posts SET embedding = %s WHERE id = %s",
                            (embedding_json, post_id)
                        )
                        
                        if (i + 1) % Config.BATCH_SIZE == 0:
                            conn.commit()
                            logger.info(f"Wrote {i + 1} embeddings...")
                    
                    conn.commit()
                    logger.info(f"Successfully wrote {len(ids)} embeddings")
                    
        except MySQLError as e:
            logger.error(f"Failed to write embeddings: {e}")
            raise

# ---------- CLUSTERING OPERATIONS ----------
class ClusteringManager:
    """Handles clustering operations"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
    
    def run_kmeans(self, k: int = Config.DEFAULT_K_CLUSTERS, visualize: bool = True):
        """Run KMeans clustering"""
        df, X = self._load_all_embeddings()
        if X.size == 0:
            logger.error("No embeddings found. Please train first.")
            return
        
        logger.info(f"Running KMeans with k={k} on {len(X)} vectors...")
        
        # Run KMeans
        kmeans = KMeans(
            n_clusters=k,
            n_init='auto',
            random_state=Config.RANDOM_STATE,
            max_iter=300,
            algorithm='lloyd'
        )
        
        labels = kmeans.fit_predict(X)
        
        # Save model
        joblib.dump(kmeans, Config.KMEANS_PATH)
        logger.info(f"Saved KMeans model to {Config.KMEANS_PATH}")
        
        # Update database
        self._update_cluster_ids(df["id"].tolist(), labels)
        
        # Visualize if requested
        if visualize:
            self._visualize_clusters(X, labels)
            
        # Generate cluster keywords
        self.generate_cluster_keywords()
    
    def _load_all_embeddings(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load all embeddings from database"""
        query = "SELECT id, embedding FROM posts WHERE embedding IS NOT NULL"
        
        try:
            with DatabaseManager.get_connection() as conn:
                df = pd.read_sql(query, conn)
                
            if df.empty:
                return df, np.array([])
                
            # Parse embeddings
            embeddings = []
            for emb_str in df["embedding"]:
                try:
                    emb = json.loads(emb_str)
                    embeddings.append(np.array(emb, dtype=np.float32))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse embedding: {e}")
                    
            X = np.vstack(embeddings) if embeddings else np.array([])
            return df, X
            
        except MySQLError as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def _update_cluster_ids(self, ids: List[int], labels: np.ndarray):
        """Update cluster IDs in database"""
        try:
            with DatabaseManager.get_connection() as conn:
                with conn.cursor() as cur:
                    for post_id, label in zip(ids, labels):
                        cur.execute(
                            "UPDATE posts SET cluster_id = %s WHERE id = %s",
                            (int(label), post_id)
                        )
                    conn.commit()
                    
            logger.info(f"Updated cluster IDs for {len(ids)} posts")
            
        except MySQLError as e:
            logger.error(f"Failed to update cluster IDs: {e}")
            raise
    
    def _visualize_clusters(self, X: np.ndarray, labels: np.ndarray):
        """Create cluster visualization"""
        logger.info("Creating cluster visualization...")
        
        # Reduce dimensionality for visualization
        reducer = umap.UMAP(
            n_components=2,
            random_state=Config.RANDOM_STATE,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine'
        )
        
        X_2d = reducer.fit_transform(X)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            X_2d[:, 0], 
            X_2d[:, 1], 
            c=labels, 
            cmap='tab10', 
            alpha=0.6, 
            s=10
        )
        plt.colorbar(scatter)
        plt.title("Reddit Posts Clustering (TF-IDF + SVD + KMeans)", fontsize=14)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = "clusters_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {output_path}")
    
    def generate_cluster_keywords(self, top_n: int = 15):
        """Generate keywords for each cluster"""
        query = """
            SELECT 
                cluster_id,
                CONCAT_WS(' ', 
                    COALESCE(title, ''), 
                    COALESCE(selftext, ''), 
                    COALESCE(ocr_text, '')
                ) AS text
            FROM posts
            WHERE cluster_id IS NOT NULL
        """
        
        try:
            with DatabaseManager.get_connection() as conn:
                df = pd.read_sql(query, conn)
                
            if df.empty:
                logger.warning("No clustered posts found")
                return
            
            # Process text and generate keywords
            stop_words = set(stopwords.words('english'))
            cluster_terms = defaultdict(Counter)
            
            for _, row in df.iterrows():
                cluster_id = row['cluster_id']
                text = TextProcessor.clean_text(row['text']).lower()
                
                # Extract words
                words = [
                    word for word in text.split()
                    if len(word) >= 3 and word not in stop_words
                ]
                
                # Count words
                cluster_terms[cluster_id].update(words)
            
            # Generate report
            results = []
            for cluster_id in sorted(cluster_terms.keys()):
                top_terms = cluster_terms[cluster_id].most_common(top_n)
                keywords = [term for term, count in top_terms]
                
                results.append({
                    'cluster_id': cluster_id,
                    'keywords': ', '.join(keywords),
                    'top_5': ', '.join(keywords[:5])
                })
            
            # Save results
            df_results = pd.DataFrame(results)
            df_results.to_csv('cluster_analysis.csv', index=False)
            logger.info("Saved cluster analysis to cluster_analysis.csv")
            
        except Exception as e:
            logger.error(f"Failed to generate cluster keywords: {e}")
            raise

# ---------- REAL-TIME OPERATIONS ----------
class RealtimeProcessor:
    """Handles real-time classification and updates"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
    
    def assign_new_posts(self, verbose: bool = True) -> int:
        """Assign clusters to new posts"""
        if not self.embedding_manager._load_models():
            logger.error("Models not found. Please train first.")
            return 0
        
        df = DatabaseManager.fetch_posts_df(only_new=True)
        if df.empty:
            if verbose:
                logger.info("No new posts to process")
            return 0
        
        # Generate embeddings
        X_tfidf = self.embedding_manager.vectorizer.transform(df["text"])
        X_embeddings = self.embedding_manager.svd.transform(X_tfidf)
        
        # Predict clusters
        labels = self.embedding_manager.kmeans.predict(X_embeddings)
        
        # Update database
        try:
            with DatabaseManager.get_connection() as conn:
                with conn.cursor() as cur:
                    for post_id, embedding, label in zip(df["id"], X_embeddings, labels):
                        embedding_json = json.dumps(embedding.tolist())
                        cur.execute(
                            "UPDATE posts SET embedding = %s, cluster_id = %s WHERE id = %s",
                            (embedding_json, int(label), post_id)
                        )
                    conn.commit()
                    
            if verbose:
                logger.info(f"Processed {len(df)} new posts")
            return len(df)
            
        except MySQLError as e:
            logger.error(f"Failed to update posts: {e}")
            raise
    
    def realtime_loop(self, interval_minutes: int = 5):
        """Run continuous processing loop"""
        logger.info(f"Starting real-time processing (interval: {interval_minutes} minutes)")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                try:
                    count = self.assign_new_posts()
                    if count > 0:
                        logger.info(f"Processed {count} new posts")
                except Exception as e:
                    logger.error(f"Error in processing loop: {e}")
                
                time.sleep(interval_minutes * 60)
                print("test")
                
        except KeyboardInterrupt:
            logger.info("Real-time processing stopped")
    
    def classify_message(self, text: str):
        """Classify a single message"""
        if not self.embedding_manager._load_models():
            logger.error("Models not found. Please train first.")
            return
        
        # Clean and process text
        clean_text = TextProcessor.clean_text(text)
        
        # Generate embedding
        X_tfidf = self.embedding_manager.vectorizer.transform([clean_text])
        X_embedding = self.embedding_manager.svd.transform(X_tfidf)[0]
        
        # Predict cluster
        cluster_id = int(self.embedding_manager.kmeans.predict([X_embedding])[0])
        
        logger.info(f"Message classified to cluster {cluster_id}")
        
        # Find similar posts
        self._find_similar_posts(cluster_id, X_embedding)
    
    def _find_similar_posts(self, cluster_id: int, query_embedding: np.ndarray, top_k: int = 5):
        """Find similar posts within the same cluster"""
        query = """
            SELECT id, title, embedding
            FROM posts
            WHERE cluster_id = %s AND embedding IS NOT NULL
            LIMIT 1000
        """
        
        try:
            with DatabaseManager.get_connection() as conn:
                df = pd.read_sql(query, conn, params=[cluster_id])
                
            if df.empty:
                logger.info("No posts found in this cluster")
                return
            
            # Calculate similarities
            embeddings = np.vstack([
                json.loads(emb) for emb in df['embedding']
            ])
            
            # Normalize vectors
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
            
            # Calculate cosine similarities
            similarities = embeddings_norm @ query_norm
            
            # Get top similar posts
            top_indices = np.argsort(-similarities)[:top_k]
            
            logger.info(f"Top {top_k} similar posts in cluster {cluster_id}:")
            for idx in top_indices:
                post_id = df.iloc[idx]['id']
                title = df.iloc[idx]['title'][:100]
                similarity = similarities[idx]
                logger.info(f"  - ID: {post_id} (similarity: {similarity:.3f}): {title}")
                
        except Exception as e:
            logger.error(f"Failed to find similar posts: {e}")

# ---------- MAIN CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Reddit Post Clustering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Initial training
    python clust_improved.py --train --kmeans 10
    
    # Update embeddings for new posts
    python clust_improved.py --update-emb
    
    # Run real-time processing
    python clust_improved.py --realtime 5
    
    # Classify a message
    python clust_improved.py --classify "What's the best Python IDE?"
        """
    )
    
    parser.add_argument(
        '--train', 
        action='store_true',
        help='Train TF-IDF and SVD models, generate embeddings'
    )
    parser.add_argument(
        '--update-emb', 
        action='store_true',
        help='Generate embeddings for posts without embeddings'
    )
    parser.add_argument(
        '--kmeans', 
        type=int, 
        metavar='K',
        help='Run KMeans clustering with K clusters'
    )
    parser.add_argument(
        '--realtime', 
        type=int, 
        metavar='MINUTES',
        help='Run real-time processing loop with specified interval'
    )
    parser.add_argument(
        '--classify', 
        type=str,
        metavar='TEXT',
        help='Classify a message into existing clusters'
    )
    parser.add_argument(
        '--keywords',
        action='store_true',
        help='Generate cluster keywords analysis'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure database schema
    try:
        DatabaseManager.ensure_columns()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return 1
    
    # Initialize managers
    embedding_manager = EmbeddingManager()
    clustering_manager = ClusteringManager(embedding_manager)
    realtime_processor = RealtimeProcessor(embedding_manager)
    
    # Execute requested operations
    try:
        if args.train:
            embedding_manager.train_tfidf_svd()
        
        if args.update_emb:
            embedding_manager.update_embeddings_only_new()
        
        if args.kmeans is not None:
            clustering_manager.run_kmeans(k=args.kmeans)
        
        if args.keywords:
            clustering_manager.generate_cluster_keywords()
        
        if args.realtime is not None:
            realtime_processor.realtime_loop(interval_minutes=args.realtime)
        
        if args.classify:
            realtime_processor.classify_message(args.classify)
            
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())