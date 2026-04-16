import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://admin:admin123@localhost:5432/phishing_db"
)


def get_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


@contextmanager
def get_db_cursor():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def init_db():
    with get_db_cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                original TEXT NOT NULL,
                suspicious TEXT NOT NULL,
                method VARCHAR(50) NOT NULL,
                is_phishing BOOLEAN NOT NULL,
                confidence REAL NOT NULL,
                label VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                prediction_id INTEGER REFERENCES predictions(id),
                url TEXT,
                email_from TEXT,
                email_subject TEXT,
                risk_score REAL,
                details JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id SERIAL PRIMARY KEY,
                original_url TEXT NOT NULL,
                suspicious_url TEXT NOT NULL,
                label VARCHAR(20) NOT NULL,
                features JSONB,
                source VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_created 
            ON predictions(created_at DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_training_data_label 
            ON training_data(label)
        """)


def save_prediction(original, suspicious, method, is_phishing, confidence, label):
    with get_db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO predictions 
            (original, suspicious, method, is_phishing, confidence, label)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (original, suspicious, method, is_phishing, confidence, label),
        )
        result = cursor.fetchone()
        return result["id"] if result else None


def get_prediction_history(limit=50):
    with get_db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id, original, suspicious, method, is_phishing, 
                   confidence, label, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return cursor.fetchall()


def get_prediction_by_id(prediction_id):
    with get_db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id, original, suspicious, method, is_phishing,
                   confidence, label, created_at
            FROM predictions
            WHERE id = %s
            """,
            (prediction_id,),
        )
        return cursor.fetchone()


def save_training_data(original_url, suspicious_url, label, features=None, source=None):
    with get_db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO training_data 
            (original_url, suspicious_url, label, features, source)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                original_url,
                suspicious_url,
                label,
                psycopg2.extras.Json(features) if features else None,
                source,
            ),
        )
        result = cursor.fetchone()
        return result["id"] if result else None


def get_training_data(label=None, limit=None):
    with get_db_cursor() as cursor:
        if label:
            query = "SELECT * FROM training_data WHERE label = %s"
            params = (label,)
        else:
            query = "SELECT * FROM training_data"
            params = None

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        return cursor.fetchall()


def get_stats():
    with get_db_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) as total FROM predictions")
        total = cursor.fetchone()["total"]

        cursor.execute(
            "SELECT COUNT(*) as count FROM predictions WHERE is_phishing = true"
        )
        phishing = cursor.fetchone()["count"]

        cursor.execute(
            "SELECT method, COUNT(*) as count FROM predictions GROUP BY method"
        )
        by_method = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) as total FROM training_data")
        training_count = cursor.fetchone()["total"]

        return {
            "total_predictions": total,
            "phishing_detected": phishing,
            "clean_detected": total - phishing,
            "by_method": by_method,
            "training_samples": training_count,
        }
