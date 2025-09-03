import os
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import Optional, Dict, Any
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CibilDatabase:
    def __init__(self):
        """Initialize database connection"""
        # For development/testing, you can use SQLite
        self.use_sqlite = os.getenv('USE_SQLITE', 'false').lower() == 'true'
        
        if self.use_sqlite:
            import sqlite3
            self.db_path = 'cibil_database.db'
            self.init_sqlite_db()
        else:
            # Production PostgreSQL configuration. Prefer DATABASE_URL from Render
            self.database_url = os.getenv('DATABASE_URL')
            if not self.database_url:
                # Fallback to discrete env vars if provided
                host = os.getenv('DB_HOST')
                user = os.getenv('DB_USER')
                password = os.getenv('DB_PASSWORD')
                dbname = os.getenv('DB_NAME')
                port = os.getenv('DB_PORT', '5432')
                if host and user and password and dbname:
                    self.database_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
            self.init_postgres_db()

    def init_sqlite_db(self):
        """Initialize SQLite database for development"""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cibil_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cibil_id TEXT UNIQUE NOT NULL,
                        cibil_score INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CHECK (cibil_score >= 300 AND cibil_score <= 900)
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_cibil_id ON cibil_scores(cibil_id)')
                
                conn.commit()
                logger.info("SQLite database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
            raise

    def init_postgres_db(self):
    """Initialize PostgreSQL database schema in production"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with psycopg2.connect(self.database_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cibil_scores (
                        id SERIAL PRIMARY KEY,
                        cibil_id TEXT UNIQUE NOT NULL,
                        cibil_score INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT cibil_score_range CHECK (cibil_score >= 300 AND cibil_score <= 900)
                    )
                    ''')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cibil_id ON cibil_scores(cibil_id)')
                    conn.commit()
                    logger.info("PostgreSQL database initialized successfully")
                    return
        except Exception as e:
            logger.error(f"Database initialization attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2)


    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        if self.use_sqlite:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
        else:
            conn = psycopg2.connect(self.database_url)
        
        try:
            yield conn
        finally:
            conn.close()

    def load_csv_to_database(self, csv_file_path: str = 'cibil_database.csv'):
        """Load CIBIL data from CSV to database"""
        # In load_csv_to_database method, ensure proper formatting:
        data_tuples = [(str(row['CIBIL ID']).strip().zfill(9), int(row['CIBIL Score']))
               for _, row in df.iterrows()]

        try:
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loading {len(df)} records from CSV file")
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if self.use_sqlite:
                    # Clear existing data
                    cursor.execute('DELETE FROM cibil_scores')
                    
                    # Bulk insert
                    data_tuples = [(str(row['CIBIL ID']).zfill(9), int(row['CIBIL Score'])) 
                                 for _, row in df.iterrows()]
                    
                    cursor.executemany(
                        'INSERT INTO cibil_scores (cibil_id, cibil_score) VALUES (?, ?)',
                        data_tuples
                    )
                else:
                    # PostgreSQL version
                    cursor.execute('DELETE FROM cibil_scores')
                    
                    from psycopg2.extras import execute_values
                    data_tuples = [(str(row['CIBIL ID']).zfill(9), int(row['CIBIL Score'])) 
                                 for _, row in df.iterrows()]
                    
                    execute_values(
                        cursor,
                        'INSERT INTO cibil_scores (cibil_id, cibil_score) VALUES %s',
                        data_tuples
                    )
                
                conn.commit()
                logger.info(f"Successfully loaded {len(df)} CIBIL records")
                
        except Exception as e:
            logger.error(f"Error loading CSV to database: {e}")
            raise

    def get_cibil_score(self, cibil_id: str) -> Optional[int]:
        """Get CIBIL score for a given CIBIL ID"""
        try:
            normalized_id = str(cibil_id).zfill(9)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if self.use_sqlite:
                    cursor.execute('SELECT cibil_score FROM cibil_scores WHERE cibil_id = ?', 
                                 (normalized_id,))
                else:
                    cursor.execute('SELECT cibil_score FROM cibil_scores WHERE cibil_id = %s', 
                                 (normalized_id,))
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Error retrieving CIBIL score: {e}")
            return None

    def validate_cibil_id(self, cibil_id: str) -> bool:
        """Validate if CIBIL ID exists in database"""
        try:
            normalized_id = str(cibil_id).zfill(9)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if self.use_sqlite:
                    cursor.execute('SELECT COUNT(*) FROM cibil_scores WHERE cibil_id = ?', 
                                 (normalized_id,))
                else:
                    cursor.execute('SELECT EXISTS(SELECT 1 FROM cibil_scores WHERE cibil_id = %s)', 
                                 (normalized_id,))
                
                result = cursor.fetchone()
                return bool(result[0])
                
        except Exception as e:
            logger.error(f"Error validating CIBIL ID: {e}")
            return False

# Global database instance
cibil_db = CibilDatabase()
