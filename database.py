import os
import time
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
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
        # For production, default to PostgreSQL unless explicitly set to use SQLite
        self.use_sqlite = os.getenv('USE_SQLITE', 'false').lower() == 'true'
        self._initialized = False
        
        if self.use_sqlite:
            import sqlite3
            self.db_path = 'cibil_database.db'
            self.init_sqlite_db()
            self._initialized = True
        else:
            # Production PostgreSQL configuration - lazy initialization
            self.database_url = os.getenv('DATABASE_URL')
            if not self.database_url:
                logger.error("DATABASE_URL environment variable is required for PostgreSQL")
                raise ValueError("DATABASE_URL environment variable is required for PostgreSQL")

    def _ensure_postgres_initialized(self):
        """Ensure PostgreSQL is initialized before use"""
        if not self._initialized and not self.use_sqlite:
            if not self.database_url:
                raise ValueError("DATABASE_URL or individual DB credentials must be provided for PostgreSQL")
            self.init_postgres_db()
            self._initialized = True

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
        """Initialize PostgreSQL database schema in production with retry logic"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting PostgreSQL connection (attempt {attempt + 1}/{max_retries})")
                with psycopg2.connect(self.database_url) as conn:
                    with conn.cursor() as cursor:
                        # Create table
                        cursor.execute('''
                        CREATE TABLE IF NOT EXISTS cibil_scores (
                            id SERIAL PRIMARY KEY,
                            cibil_id TEXT UNIQUE NOT NULL,
                            cibil_score INTEGER NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            CONSTRAINT cibil_score_range CHECK (cibil_score >= 300 AND cibil_score <= 900)
                        )
                        ''')
                        
                        # Create indexes
                        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cibil_id ON cibil_scores(cibil_id)')
                        conn.commit()
                        logger.info("PostgreSQL database initialized successfully")
                        return
                        
            except psycopg2.OperationalError as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("All database connection attempts failed")
                    raise
                time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Error initializing PostgreSQL database: {e}")
                raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            if self.use_sqlite:
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
            else:
                self._ensure_postgres_initialized()
                conn = psycopg2.connect(self.database_url)
            yield conn
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def get_record_count(self) -> int:
        """Get the total number of records in the database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM cibil_scores')
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting record count: {e}")
            return 0

    def load_csv_to_database(self, csv_file_path: str = 'cibil_database.csv'):
        """Load CIBIL data from CSV to database with improved error handling"""
        try:
            # Check if file exists
            if not os.path.exists(csv_file_path):
                logger.error(f"CSV file not found: {csv_file_path}")
                raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
            
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loading {len(df)} records from CSV file: {csv_file_path}")
            
            # Validate required columns
            required_columns = ['CIBIL ID', 'CIBIL Score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")
            
            # Clean and validate data
            df = df.dropna(subset=required_columns)  # Remove rows with missing data
            df = df[df['CIBIL Score'].between(300, 900)]  # Filter valid scores
            
            logger.info(f"After cleaning: {len(df)} valid records to load")
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if self.use_sqlite:
                    # Clear existing data
                    cursor.execute('DELETE FROM cibil_scores')
                    
                    # Prepare data with proper formatting
                    data_tuples = [(str(row['CIBIL ID']).strip().zfill(9), int(row['CIBIL Score']))
                                 for _, row in df.iterrows()]
                    
                    # Bulk insert
                    cursor.executemany(
                        'INSERT INTO cibil_scores (cibil_id, cibil_score) VALUES (?, ?)',
                        data_tuples
                    )
                else:
                    # Clear existing data
                    cursor.execute('DELETE FROM cibil_scores')
                    
                    # Prepare data with proper formatting
                    data_tuples = [(str(row['CIBIL ID']).strip().zfill(9), int(row['CIBIL Score']))
                                 for _, row in df.iterrows()]
                    
                    # Use execute_values for efficient bulk insert
                    execute_values(
                        cursor,
                        'INSERT INTO cibil_scores (cibil_id, cibil_score) VALUES %s',
                        data_tuples,
                        template=None,
                        page_size=1000  # Process in batches
                    )
                
                conn.commit()
                logger.info(f"Successfully loaded {len(df)} CIBIL records into database")
                
        except Exception as e:
            logger.error(f"Error loading CSV to database: {e}")
            raise

    def get_cibil_score(self, cibil_id: str) -> Optional[int]:
        """Get CIBIL score for a given CIBIL ID"""
        try:
            normalized_id = str(cibil_id).strip().zfill(9)
            
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
            logger.error(f"Error retrieving CIBIL score for ID {cibil_id}: {e}")
            return None

    def validate_cibil_id(self, cibil_id: str) -> bool:
        """Validate if CIBIL ID exists in database"""
        try:
            normalized_id = str(cibil_id).strip().zfill(9)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if self.use_sqlite:
                    cursor.execute('SELECT COUNT(*) FROM cibil_scores WHERE cibil_id = ?',
                                 (normalized_id,))
                    result = cursor.fetchone()
                    return bool(result[0]) if result else False
                else:
                    cursor.execute('SELECT EXISTS(SELECT 1 FROM cibil_scores WHERE cibil_id = %s)',
                                 (normalized_id,))
                    result = cursor.fetchone()
                    return bool(result[0]) if result else False
                    
        except Exception as e:
            logger.error(f"Error validating CIBIL ID {cibil_id}: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM cibil_scores')
                record_count = cursor.fetchone()[0]
                
                return {
                    "database_connected": True,
                    "database_type": "SQLite" if self.use_sqlite else "PostgreSQL",
                    "record_count": record_count,
                    "status": "healthy"
                }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "database_connected": False,
                "database_type": "SQLite" if self.use_sqlite else "PostgreSQL",
                "record_count": 0,
                "status": "unhealthy",
                "error": str(e)
            }

# Global database instance
cibil_db = CibilDatabase()
