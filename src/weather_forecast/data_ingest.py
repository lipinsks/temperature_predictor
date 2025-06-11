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
    """Wyjątek zgłaszany przy problemach z load()."""
    pass

# --- Abstrakcja ----------------------------------------------------------------

class BaseDataSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Ładuje dane i zwraca pandas.DataFrame z kolumnami:
        ['id','temperature','lat','lon','country_code','city','timestamp'].
        """
        ...

# --- Implementacje -----------------------------------------------------------

class SQLiteSource(BaseDataSource):
    def __init__(self, db_path: str, table: Optional[str] = None, query: Optional[str] = None):
        """
        db_path: ścieżka do pliku bazy SQLite (.db / .sqlite3)
        table: nazwa tabeli do wczytania. Jeśli podane, ignoruje query.
        query: dowolne zapytanie SQL.
        """
        self.db_path = db_path
        self.table = table
        self.query = query
        conn_str = f"sqlite:///{self.db_path}"
        self.engine = create_engine(conn_str)

    def load(self) -> pd.DataFrame:
        logger.info("Ładuję dane z SQLite: %s", self.db_path)
        try:
            if self.table:
                q = f"SELECT * FROM {self.table}"
            elif self.query:
                q = self.query
            else:
                raise DataSourceError("Musisz podać table lub query dla SQLiteSource")
            df = pd.read_sql(q, self.engine)
            _validate_schema(df)
            return df
        except Exception as e:
            raise DataSourceError(f"SQLite load failed: {e}")

class PostgresSource(BaseDataSource):
    def __init__(self, conn_str: str, query: str, connect_args: Optional[Dict[str,Any]] = None):
        self.engine = create_engine(conn_str, connect_args=connect_args or {})
        self.query = query

    def load(self) -> pd.DataFrame:
        logger.info("Ładuję dane z Postgresa")
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
        logger.info("Ładuję dane z CSV: %s", self.filepath)
        try:
            df = pd.read_csv(self.filepath)
            _validate_schema(df)
            return df
        except Exception as e:
            raise DataSourceError(f"CSV load failed: {e}")

class S3Source(BaseDataSource):
    def __init__(self, bucket: str, key: str, aws_cfg: Optional[Dict[str,Any]] = None):
        self.bucket = bucket
        self.key = key
        self.s3 = boto3.client("s3", **(aws_cfg or {}))

    def load(self) -> pd.DataFrame:
        logger.info("Ładuję dane z S3: s3://%s/%s", self.bucket, self.key)
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.key)
            df = pd.read_csv(obj["Body"])
            _validate_schema(df)
            return df
        except (BotoCoreError, ClientError, Exception) as e:
            raise DataSourceError(f"S3 load failed: {e}")

# --- Walidacja i utils -------------------------------------------------------

def _validate_schema(df: pd.DataFrame) -> None:
    required = {"id", "temperature", "lat", "lon", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise DataSourceError(f"Brakuje kolumn: {missing}")
    # upewnij się, że timestamp jest datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")



# EXAMPLE USAGE
def example_usage():
    db_file = "data/weather.db"

    try:
        src = SQLiteSource(
            db_path=db_file,
            table="weather"
        )
        df = src.load()
        print("Załadowano rekordów:", len(df))
        print(df.head())
    except DataSourceError as e:
        print("Błąd wczytywania:", e)
