import os
import io
import datetime
import tempfile
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple, Any
import sqlite3
try:
    from PyQt5.QtGui import QStandardItemModel, QStandardItem
    HAVE_PYQT = True
except Exception:
    HAVE_PYQT = False
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    import pydicom
    from pydicom.filereader import dcmread
    HAVE_PYDICOM = True
except Exception:
    HAVE_PYDICOM = False

try:
    import nibabel as nib
    HAVE_NIBABEL = True
except Exception:
    HAVE_NIBABEL = False

try:
    from pymongo import MongoClient
    HAVE_PYMONGO = True
except Exception:
    HAVE_PYMONGO = False
    

class SignalController:
    def __init__(self, view, model):  # view: SignalWidget, model: SignalProcessor
        self.view = view
        self.model = model

    # La vista llamará este método para cargar una señal.
    def handle_load_signal(self):
        file_path = self.view.get_selected_file()
        if not file_path:
            return False

        # Por ahora asumimos que es un .mat
        self.model.load_mat_file(file_path)
        self.model.compute_fft_all_channels()

        # Llenar la tabla con los resultados de la FFT
        df = self.model.get_fft_dataframe()
        self.view.populate_table(df)

        # Llenar el combo de canales
        self.view.channel_combo.clear()
        channels = sorted(self.model.signal_data.keys())
        for ch in channels:
            self.view.channel_combo.addItem(str(ch))

        return True

    # La vista pide un gráfico espectral.
    def handle_plot_spectrum(self):
        channel_index = self.view.get_selected_channel_index()
        fig = self.model.get_spectrum_plot(channel_index)
        return fig

    # La vista pide desviación estándar + histograma.
    def handle_std_dev(self):
        std_value, fig = self.model.calculate_std_and_histogram(axis="global")
        return std_value, fig


class TabularProcessor:
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None

    def load_csv(self, file_path: str, **kwargs):
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        self.data = pd.read_csv(file_path, **kwargs)
        return self.data

    def get_columns(self) -> List[str]:
        if self.data is None:
            return []
        return list(self.data.columns)

    def get_column_plot(self, column: str) -> plt.Figure:
        if self.data is None:
            raise RuntimeError("No hay datos cargados")
        if column not in self.data.columns:
            raise KeyError(column)
        fig, ax = plt.subplots()
        ax.plot(self.data.index, self.data[column])
        ax.set_title(column)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        fig.tight_layout()
        return fig

    def get_data_model(self):
        if not HAVE_PYQT:
            raise RuntimeError("PyQt5 no disponible para crear QStandardItemModel.")
        if self.data is None:
            raise RuntimeError("No hay datos cargados.")
        model = QStandardItemModel()
        # set headers
        headers = list(self.data.columns)
        model.setColumnCount(len(headers))
        model.setRowCount(len(self.data))
        model.setHorizontalHeaderLabels(headers)
        for r in range(len(self.data)):
            for c, col in enumerate(headers):
                val = self.data.iloc[r, c]
                item = QStandardItem(str(val))
                model.setItem(r, c, item)
        return model
class DatabaseLogger:
    def __init__(self, db_type: str = "sqlite", credentials: Optional[Dict[str, Any]] = None):
        self.db_type = db_type.lower()
        self.credentials = credentials or {}
        self.conn = None
        self.client = None
        if self.db_type == "sqlite":
            db_path = self.credentials.get('path', 'pmda_activity.db')
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self._ensure_table_sqlite()
        elif self.db_type == "mongo":
            if not HAVE_PYMONGO:
                raise RuntimeError("pymongo no disponible para usar MongoDB.")
            uri = self.credentials.get('uri', 'mongodb://localhost:27017')
            dbname = self.credentials.get('db', 'pmda')
            coll = self.credentials.get('collection', 'activity')
            self.client = MongoClient(uri)
            self.collection = self.client[dbname][coll]
        else:
            raise ValueError("db_type debe ser 'sqlite' o 'mongo'")

    def _ensure_table_sqlite(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                action TEXT,
                timestamp TEXT,
                result_path TEXT
            )
        """)
        self.conn.commit()

    def log_activity(self, user: str, action: str, result_path: Optional[str] = None):
        ts = datetime.datetime.now().isoformat()
        if self.db_type == "sqlite":
            cur = self.conn.cursor()
            cur.execute("INSERT INTO activity (username, action, timestamp, result_path) VALUES (?, ?, ?, ?)",
                        (user, action, ts, result_path or ""))
            self.conn.commit()
        else:
            doc = {"username": user, "action": action, "timestamp": ts, "result_path": result_path or ""}
            self.collection.insert_one(doc)

    def close(self):
        if self.conn:
            self.conn.close()
        if self.client:
            self.client.close()

    def save_array_as_png(array: np.ndarray, out_path: str):
        """Guarda array 2D como PNG (uint8)."""
        if not HAVE_CV2:
            raise RuntimeError("OpenCV no disponible.")
        arr = array.copy()
        if arr.dtype != np.uint8:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, arr)