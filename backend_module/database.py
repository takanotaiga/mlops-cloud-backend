from typing import Dict, List
from surrealdb import Surreal
import threading
import time

class DataBaseManager:
    def __init__(
            self,
            endpoint_url: str,
            username: str,
            password: str,
            namespace: str,
            database: str
        ):
        self._surreal_client = Surreal(endpoint_url)
        for _ in range(5):
            try:
                self._surreal_client.signin({"username":username, "password":password})
                self._surreal_client.use(namespace, database)
                break
            except ConnectionRefusedError:
                time.sleep(1)
        else:
            raise RuntimeError("Failed to connect to SurrealDB")

        self._db_lock = threading.Lock()

    def query(self, query: str, vars: Dict = None) -> (List[dict] | dict):
        with self._db_lock:
            return self._surreal_client.query(query, vars)
