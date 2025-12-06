import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pydicom
import nibabel as nib
import scipy.io as sio
import pandas as pd
from datetime import datetime
import pymongo  
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import io
import datetime
import tempfile
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import os
from typing import Dict, Optional, List, Tuple, Any
import sqlite3

class AuthManager:
    def __init__(self, xml_path="config/users.xml"):
        self.xml_path = xml_path       
        self.users = {}                
        self._cargar_usuarios()         

    def _cargar_usuarios(self):
        #
        if not os.path.exists(self.xml_path):
            os.makedirs("config", exist_ok=True)
            root = ET.Element("usuarios")
            ET.SubElement(root, "usuario", nombre="admin", contrasena="1234")
            tree = ET.ElementTree(root)
            tree.write(self.xml_path)
            print(f"[AuthManager] Creado {self.xml_path} con usuarios de prueba")
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            for usuario in root.findall("usuario"):
                nombre = usuario.get("nombre")
                contrasena = usuario.get("contrasena")
                if nombre and contrasena:
                    self.users[nombre] = contrasena
        except Exception as e:
            print(f"Error leyendo XML: {e}")
            self.users["admin"] = "1234"

    def verify_credentials(self, username, password):
        return self.users.get(username) == password

class UserSession:
    def __init__(self, username: str):
        self.username = username                    
        self.login_time = datetime.now()          
        self.temp_folder = self.create_temp_folder()

    def create_temp_folder(self) -> str:
        folder_path = f"temp/{self.username}"
        os.makedirs(folder_path, exist_ok=True)
        print(f"[UserSession] Carpeta temporal creada: {folder_path}")
        return folder_path

    def get_username(self):
        return self.username

    def get_user_info(self):
        duracion = datetime.now() - self.login_time
        info = {
            "username": self.username,
            "login_time": self.login_time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_minutes": round(duracion.total_seconds() / 60, 2),
            "temp_folder": self.temp_folder
        }
        return info
    
class ImageProcessor:
    def __init__(self):
        self.volume = None              
        self.metadata = {}
        self.current_file = ""

    def load_image(self, path: str) -> bool:    
        self.volume = None
        self.metadata = {}
        self.current_slices = {}
        
        ext = os.path.splitext(path.lower())[1]

        try:
            if ext == ".dcm":
                ds = pydicom.dcmread(path)
                self.volume = ds.pixel_array.astype(np.float32)
                self.metadata = {
                    "PatientName": str(ds.get("PatientName", "Anónimo")),
                    "PatientID": str(ds.get("PatientID", "")),
                    "StudyDate": str(ds.get("StudyDate", "")),
                    "Modality": ds.get("Modality", "")
                }
    
                if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                    self.volume = self.volume * ds.RescaleSlope + ds.RescaleIntercept

            elif ext in [".nii", ".gz"]:
                img = nib.load(path)
                self.volume = np.asanyarray(img.dataobj)
                self.metadata = {"Modality": "MRI", "PatientName": "NIFTI"}

            else:  
                img = cv2.imread(path)
                if img is None:
                    return False
                self.volume = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.metadata = {"Modality": "2D", "PatientName": os.path.basename(path)}

           
            self._save_metadata_to_csv(path)
            return True

        except Exception as e:
            print(f"Error cargando imagen: {e}")
            return False

    def _save_metadata_csv(self):
        df = pd.DataFrame([self.metadata])
        csv_path = f"resultados/metadata_{os.path.basename(self.current_file)}.csv"
        os.makedirs("resultados", exist_ok=True)
        df.to_csv(csv_path, index=False)

    def get_slice(self, plane: str, index: int) -> QPixmap:
        if self.volume is None:
            return QPixmap()

        if self.file_type == "2d":
            img = self.volume
        else:
            if plane == "axial":
                img = self.volume[:, :, index] if self.volume.ndim == 3 else self.volume[index]
            elif plane == "coronal":
                img = self.volume[:, index, :]
            elif plane == "sagittal":
                img = self.volume[index, :, :]
            else:
                return QPixmap()

      
        img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)

        if img.ndim == 2:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)

        return QPixmap.fromImage(qimg)

    def get_max_slices(self, plane: str) -> int:
        if self.volume is None:
            return 0
        if self.file_type == "2d":
            return 0
        if plane == "axial":
            return self.volume.shape[2] - 1
        elif plane == "coronal":
            return self.volume.shape[1] - 1
        elif plane == "sagittal":
            return self.volume.shape[0] - 1
        return 0

    def apply_filter(self, filter_type: str, image_2d: np.ndarray) -> QPixmap:
    
        gray = cv2.cvtColor(image_2d, cv2.COLOR_RGB2GRAY) if image_2d.ndim == 3 else image_2d

        if filter_type == "grayscale":
            pass
        elif filter_type == "gaussian":
            gray = cv2.GaussianBlur(gray, (5,5), 0)
        elif filter_type == "binarizacion":
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        elif filter_type == "canny":
            gray = cv2.Canny(gray, 100, 200)
        elif filter_type == "equalize":
            gray = cv2.equalizeHist(gray)

        h, w = gray.shape
        qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

    def convert_to_grayscale_and_save(self, frame: np.ndarray, username: str) -> str:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        path = UserSession().get_temp_path(f"perfil_{username}.jpg")
        cv2.imwrite(path, gray)
        return path
    
try:
    from PyQt5.QtGui import QStandardItemModel, QStandardItem
    HAVE_PYQT = True
except Exception:
    HAVE_PYQT = False 
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