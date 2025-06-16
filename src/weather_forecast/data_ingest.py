# src/weather_forecast/data_ingest.py

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
from sqlalchemy import create_engine
import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# --- Exceptions --------------------------------------------------------------

class DataSourceError(Exception):
    """Raised when load() fails for any data source."""
    pass

# --- Abstraction -------------------------------------------------------------

class BaseDataSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load data and return a DataFrame with at least:
        ['id','temperature','lat','lon','timestamp'] plus our new API columns.
        """
        ...

# --- Implementations --------------------------------------------------------

class SQLiteSource(BaseDataSource):
    def __init__(self, db_path: str, table: Optional[str] = None, query: Optional[str] = None):
        """
        db_path: path to SQLite file
        table: table name to read (ignores query if set)
        query: custom SQL query
        """
        self.db_path = db_path
        self.table   = table
        self.query   = query
        self.engine  = create_engine(f"sqlite:///{db_path}")

    def load(self) -> pd.DataFrame:
        logger.info("Loading data from SQLite: %s", self.db_path)
        try:
            if self.table:
                q = f"SELECT * FROM {self.table}"
            elif self.query:
                q = self.query
            else:
                raise DataSourceError("Must provide table or query for SQLiteSource")
            df = pd.read_sql(q, self.engine)
            _validate_schema(df)
            return df
        except Exception as e:
            raise DataSourceError(f"SQLite load failed: {e}")

class PostgresSource(BaseDataSource):
    def __init__(self, conn_str: str, query: str, connect_args: Optional[Dict[str,Any]] = None):
        self.engine = create_engine(conn_str, connect_args=connect_args or {})
        self.query  = query

    def load(self) -> pd.DataFrame:
        logger.info("Loading data from Postgres")
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(self.query, conn)
            _validate_schema(df)
            return df
        except Exception as e:
            raise DataSourceError(f"Postgres load failed: {e}")

class CSVSource(BaseDataSource):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        logger.info("Loading data from CSV: %s", self.filepath)
        try:
            df = pd.read_csv(self.filepath)
            _validate_schema(df)
            return df
        except Exception as e:
            raise DataSourceError(f"CSV load failed: {e}")

class S3Source(BaseDataSource):
    def __init__(self, bucket: str, key: str, aws_cfg: Optional[Dict[str,Any]] = None):
        self.bucket = bucket
        self.key    = key
        self.s3     = boto3.client("s3", **(aws_cfg or {}))

    def load(self) -> pd.DataFrame:
        logger.info("Loading data from S3: s3://%s/%s", self.bucket, self.key)
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.key)
            df  = pd.read_csv(obj["Body"])
            _validate_schema(df)
            return df
        except (BotoCoreError, ClientError, Exception) as e:
            raise DataSourceError(f"S3 load failed: {e}")

# --- Schema validation ------------------------------------------------------

def _validate_schema(df: pd.DataFrame) -> None:
    # We now require the core API fields plus the original ones
    required = {
        "id", "temperature", "lat", "lon", "timestamp",
        "feels_like", "temp_min", "temp_max",
        "pressure", "humidity", "visibility",
        "wind_speed", "wind_deg", "clouds_all",
        "weather_main"
    }
    missing = required - set(df.columns)
    if missing:
        raise DataSourceError(f"Missing columns: {missing}")
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")


# --- Example usage ----------------------------------------------------------

def example_usage():
    db_file = "data/weather.db"
    try:
        src = SQLiteSource(db_path=db_file, table="weather")
        df  = src.load()
        print("Loaded rows:", len(df))
        print(df.head())
    except DataSourceError as e:
        print("Data ingest error:", e)
