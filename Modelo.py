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
    
class SignalProcessor:
    def __init__(self):
        self.signal_data: Dict[int, np.ndarray] = {}  
        self.sampling_rate: Optional[float] = None
        self.fft_results: Optional[pd.DataFrame] = None

    def load_mat_file(self, path: str, variable_name: Optional[str] = None) -> bool:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        mat = loadmat(path)
        keys = [k for k in mat if not k.startswith('__')]
        if not keys:
            raise RuntimeError("No se encontraron variables en el .mat")

        data = None
        if variable_name and variable_name in mat:
            data = mat[variable_name]
        else:
            data = max((mat[k] for k in keys), key=lambda x: np.size(x))
        arr = np.array(data)
        if arr.ndim == 1:
            self.signal_data[0] = arr.astype(np.float32)
        elif arr.ndim == 2:
            if arr.shape[0] < arr.shape[1]:
                if arr.shape[0] <= 16:
                    self.signal_data = {i: arr[i, :].astype(np.float32) for i in range(arr.shape[0])}
                else:
                    self.signal_data = {i: arr[:, i].astype(np.float32) for i in range(arr.shape[1])}
            else:
                self.signal_data = {i: arr[i, :].astype(np.float32) for i in range(arr.shape[0])}
        else:
            shape = arr.shape
            channels = shape[0]
            self.signal_data = {i: arr[i, :].astype(np.float32) for i in range(channels)}

        if 'fs' in mat:
            try:
                self.sampling_rate = float(mat['fs'].squeeze())
            except Exception:
                self.sampling_rate = None
        else:
            self.sampling_rate = mat.get('sampling_rate', None)
            if isinstance(self.sampling_rate, np.ndarray):
                self.sampling_rate = float(np.squeeze(self.sampling_rate))
        return True

    def compute_fft_all_channels(self):
        if not self.signal_data:
            raise RuntimeError("No hay señales cargadas.")
        rows = []
        for ch, sig in self.signal_data.items():
            n = sig.size
            yf = np.abs(rfft(sig))
            if self.sampling_rate is None:
                xf = rfftfreq(n, d=1.0)
            else:
                xf = rfftfreq(n, d=1.0 / float(self.sampling_rate))
            idx = np.argmax(yf)
            dominant_freq = float(xf[idx])
            dominant_mag = float(yf[idx])
            rows.append({
                'channel': int(ch),
                'dominant_freq': dominant_freq,
                'dominant_mag': dominant_mag,
                'freqs': xf.tolist(),
                'mags': yf.tolist()
            })
        self.fft_results = pd.DataFrame(rows)
        return self.fft_results

    def get_fft_dataframe(self) -> pd.DataFrame:
        if self.fft_results is None:
            raise RuntimeError("FFT no calculada aún.")
        return self.fft_results

    def save_fft_csv(self, out_path: str) -> str:
        if self.fft_results is None:
            raise RuntimeError("FFT no calculada.")
        df = self.fft_results.copy()
        df['freqs'] = df['freqs'].apply(lambda x: ",".join(map(str, x)))
        df['mags'] = df['mags'].apply(lambda x: ",".join(map(str, x)))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path

    def get_spectrum_plot(self, channel: int) -> plt.Figure:
        if channel not in self.signal_data:
            raise IndexError("Canal no encontrado")
        sig = self.signal_data[channel]
        n = sig.size
        yf = np.abs(rfft(sig))
        if self.sampling_rate is None:
            xf = rfftfreq(n, d=1.0)
        else:
            xf = rfftfreq(n, d=1.0 / float(self.sampling_rate))
        fig, ax = plt.subplots()
        ax.plot(xf, yf)
        ax.set_title(f"Spectrum - Channel {channel}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(True)
        fig.tight_layout()
        return fig

    def calculate_std_and_histogram(self, axis: str = 'channel') -> Tuple[float, plt.Figure]:
        if not self.signal_data:
            raise RuntimeError("No hay señal cargada.")
        if axis == 'channel':
            stds = [float(np.std(sig)) for sig in self.signal_data.values()]
            std_value = float(np.mean(stds))
            data_for_hist = stds
            title = "Histogram of channel std devs"
        elif axis == 'global':
            concat = np.hstack([sig for sig in self.signal_data.values()])
            std_value = float(np.std(concat))
            data_for_hist = concat
            title = "Histogram of global samples"
        else:
            raise ValueError("axis debe ser 'channel' o 'global'")

        fig, ax = plt.subplots()
        ax.hist(data_for_hist, bins=30)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        fig.tight_layout()
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