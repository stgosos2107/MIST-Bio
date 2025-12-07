# MIST-Bio
# Mariana Ardila Alvarez
# Sofia Henao Osorio
# Valeria Salazar Ibarguen
# Santiago Osorio Salazar
#Este .py contiene las clases del modelo para manejar datos y lógica del aplicativo MIST-Bio.

#Importaciones necesarias 
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.fft import rfft, rfftfreq
import sqlite3
from PyQt5.QtGui import (
    QPixmap,
    QImage,
    QStandardItemModel,
    QStandardItem,
)
from PyQt5.QtCore import Qt

HAVE_PYQT = True


try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    import pydicom
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


#CLASES DEL MODELO

#Esta primer clase se encarga del login y registro de usuarios
class AuthManager:
    def __init__(self, xml_path: str = "config/users.xml"):
        self.xml_path = xml_path
        self.users: Dict[str, str] = {}
        self._cargar_usuarios()

#metodo privado para cargar usuarios desde el XML
    def _cargar_usuarios(self):
        if not os.path.exists(self.xml_path):
            os.makedirs("config", exist_ok=True)
            root = ET.Element("usuarios")
            ET.SubElement(root, "usuario", nombre="admin", contrasena="1234")
            tree = ET.ElementTree(root)
            tree.write(self.xml_path)

        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            for usuario in root.findall("usuario"):
                nombre = usuario.get("nombre")
                contrasena = usuario.get("contrasena")
                if nombre and contrasena:
                    self.users[nombre] = contrasena
        except Exception:
            self.users["admin"] = "1234"

#método para verificar credenciales
    def verify_credentials(self, username, password):
        return self.users.get(username) == password

#método para registrar un nuevo usuario
    def register_user(self, username, password):
        username = (username or "").strip()
        if not username or not password:
            return False

        if username in self.users:
            return False

        self.users[username] = password

        if os.path.exists(self.xml_path):
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
        else:
            os.makedirs(os.path.dirname(self.xml_path) or ".", exist_ok=True)
            root = ET.Element("usuarios")
            tree = ET.ElementTree(root)

        ET.SubElement(root, "usuario", nombre=username, contrasena=password)
        tree.write(self.xml_path)

        return True


#Esta segunda clase gestiona la sesión del usuario
class UserSession:
    def __init__(self, username):
        self.username = username
        self.login_time = datetime.now()
        self.temp_folder = self.create_temp_folder()

#Este método crea y retorna la ruta donde se guardarán archivos temporales del usuario
    def create_temp_folder(self):
        folder_path = os.path.join("temp", self.username)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

#método que consigue el nombre del usuario
    def get_username(self):
        return self.username

#método que consigue el nombre del usuario
    def get_user(self):
        return self.username

#método que genera un reporte completo de la sesión del usuario.
    def get_user_info(self) -> Dict[str, Any]:
        duracion = datetime.now() - self.login_time
        info = {
            "username": self.username,
            "login_time": self.login_time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_minutes": round(duracion.total_seconds() / 60, 2),
            "temp_folder": self.temp_folder,
        }
        return info

# Este método devuelve los datos finales de la sesión para que el MainController registre esa actividad.
    def end_session(self) -> Dict[str, Any]:
        return self.get_user_info()

#el ultimo método de esta clase sirve para guardar o recuperar archivos temporales dentro de la carpeta del usuario.
    def get_temp_path(self, filename):
        return os.path.join(self.temp_folder, filename)


#La tercer clase se encarga de procesar imágenes médicas
class ImageProcessor:
    def __init__(self):
        self.volume: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}
        self.current_file: str = ""
        self.file_type: str = "2d"          # "2d" o "3d"
        # (row_spacing, col_spacing, slice_thickness)
        self.spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)


#método que identifica el tipo de imagen y la carga correctamente
    def load_image(self, path):
        self.volume = None
        self.metadata = {}
        self.current_file = path
        self.file_type = "2d"
        self.spacing = (1.0, 1.0, 1.0)

        try:
            if os.path.isdir(path):
# Carpeta con una serie DICOM
                return self._load_dicom_series(path)

            ext = os.path.splitext(path.lower())[1]

            if ext == ".dcm":
                return self._load_single_dicom(path)
            elif ext in (".nii", ".gz"):
                return self._load_nifti(path)
            else:
                return self._load_2d_image(path)

        except Exception as e:
            print(f"[ImageProcessor] Error cargando imagen: {e}")
            return False


#método que construye un volumen 3D a partir de una carpeta con cortes DICOM.
    def _load_dicom_series(self, folder):
        if not HAVE_PYDICOM:
            raise RuntimeError("pydicom no disponible.")

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".dcm")
        ]
        if not files:
            raise RuntimeError("No se encontraron archivos DICOM en la carpeta.")

        datasets = []
        for fp in files:
            try:
                ds = pydicom.dcmread(fp)
                datasets.append(ds)
            except Exception:
                pass

        if not datasets:
            raise RuntimeError("No se pudieron leer archivos DICOM.")


        def sort_key(ds):
            z = None
            if hasattr(ds, "ImagePositionPatient"):
                try:
                    z = float(ds.ImagePositionPatient[2])
                except Exception:
                    z = None
            if z is None:
                try:
                    z = float(getattr(ds, "InstanceNumber", 0))
                except Exception:
                    z = 0.0
            return z

        datasets.sort(key=sort_key)

        arrs = [ds.pixel_array.astype(np.float32) for ds in datasets]
        vol = np.stack(arrs, axis=-1)  # (rows, cols, slices)

        self.volume = vol
        self.file_type = "3d"

        first = datasets[0]

        row_sp, col_sp = 1.0, 1.0
        if hasattr(first, "PixelSpacing"):
            try:
                row_sp, col_sp = [float(x) for x in first.PixelSpacing]
            except Exception:
                pass

        slice_th = 1.0
        if hasattr(first, "SliceThickness"):
            try:
                slice_th = float(first.SliceThickness)
            except Exception:
                pass
        elif hasattr(first, "SpacingBetweenSlices"):
            try:
                slice_th = float(first.SpacingBetweenSlices)
            except Exception:
                pass

        self.spacing = (row_sp, col_sp, slice_th)

        self.metadata = {
            "PatientName": str(first.get("PatientName", "Anónimo")),
            "PatientID": str(first.get("PatientID", "")),
            "StudyDate": str(first.get("StudyDate", "")),
            "Modality": str(first.get("Modality", "")),
            "NumSlices": vol.shape[2],
        }

        self._save_metadata_csv()
        return True

#Este método carga un archivo DICOM individual y extrae datos importantes.
    def _load_single_dicom(self, path):
        if not HAVE_PYDICOM:
            raise RuntimeError("pydicom no disponible.")

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        self.volume = arr
        self.file_type = "3d" if arr.ndim == 3 else "2d"

        row_sp, col_sp = 1.0, 1.0
        if hasattr(ds, "PixelSpacing"):
            try:
                row_sp, col_sp = [float(x) for x in ds.PixelSpacing]
            except Exception:
                pass

        slice_th = 1.0
        if hasattr(ds, "SliceThickness"):
            try:
                slice_th = float(ds.SliceThickness)
            except Exception:
                pass
        elif hasattr(ds, "SpacingBetweenSlices"):
            try:
                slice_th = float(ds.SpacingBetweenSlices)
            except Exception:
                pass

        self.spacing = (row_sp, col_sp, slice_th)

        self.metadata = {
            "PatientName": str(ds.get("PatientName", "Anónimo")),
            "PatientID": str(ds.get("PatientID", "")),
            "StudyDate": str(ds.get("StudyDate", "")),
            "Modality": str(ds.get("Modality", "")),
            "NumSlices": arr.shape[2] if arr.ndim == 3 else 1,
        }

        self._save_metadata_csv()
        return True

#método que carga archivos NIfTI y su información relevante.
    def _load_nifti(self, path):
        if not HAVE_NIBABEL:
            raise RuntimeError("nibabel no disponible.")

        img = nib.load(path)
        arr = np.asanyarray(img.dataobj).astype(np.float32)
        self.volume = arr
        self.file_type = "3d" if arr.ndim == 3 else "2d"

        try:
            zooms = img.header.get_zooms()
            if len(zooms) >= 3:
                self.spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
        except Exception:
            self.spacing = (1.0, 1.0, 1.0)

        self.metadata = {
            "Modality": "MRI",
            "PatientName": os.path.basename(path),
        }
        self._save_metadata_csv()
        return True

#Carga imágenes comunes 2D como fotografías o radiografías en PNG/JPG
    def _load_2d_image(self, path):
        if not HAVE_CV2:
            raise RuntimeError("OpenCV no disponible.")
        img = cv2.imread(path)
        if img is None:
            return False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.volume = img.astype(np.float32)
        self.file_type = "2d"
        self.spacing = (1.0, 1.0, 1.0)
        self.metadata = {
            "Modality": "2D",
            "PatientName": os.path.basename(path),
        }
        self._save_metadata_csv()
        return True

#metodo que crea automáticamente un archivo CSV con la información de la imagen cargada
    def _save_metadata_csv(self):
        if not self.metadata:
            return
        df = pd.DataFrame([self.metadata])
        os.makedirs("resultados", exist_ok=True)
        base = os.path.basename(self.current_file) or "imagen"
        csv_path = os.path.join("resultados", f"metadata_{base}.csv")
        df.to_csv(csv_path, index=False)

#Extrae un corte 2D desde un volumen 3D para mostrarlo en pantalla.
    def get_slice(self, plane, index):
        if self.volume is None or not HAVE_PYQT:
            return QPixmap()

        vol = self.volume
        plane = plane.lower()

        # Elegir el array 2D
        if vol.ndim == 2 or self.file_type == "2d":
            arr = vol
        else:
            if plane == "axial":
                index = max(0, min(index, vol.shape[2] - 1))
                arr = vol[:, :, index]
            elif plane == "coronal":
                index = max(0, min(index, vol.shape[1] - 1))
                arr = vol[:, index, :]
            elif plane == "sagittal":
                index = max(0, min(index, vol.shape[0] - 1))
                arr = vol[index, :, :]
            else:
                index = max(0, min(index, vol.shape[2] - 1))
                arr = vol[:, :, index]

        arr = np.array(arr, dtype=np.float32)

# Ventana de intensidades
        lo = np.percentile(arr, 1)
        hi = np.percentile(arr, 99)
        arr = np.clip(arr, lo, hi)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        img = (arr * 255).astype(np.uint8)

# Corrección de aspecto y orientación para coronal/sagittal 3D
        if self.file_type == "3d" and vol.ndim == 3 and plane in ("coronal", "sagittal"):
            row_sp, col_sp, slice_th = self.spacing
            row_sp = row_sp or 1.0
            col_sp = col_sp or 1.0
            slice_th = slice_th or 1.0

# Para coronal uso spacing de filas, para sagital el de columnas
            base_spacing = row_sp if plane == "coronal" else col_sp

# Factor físico: qué tanto más “gordo” debería ser el eje de los cortes
            phys_factor = slice_th / base_spacing if base_spacing > 0 else 1.0
# Lo usamos casi tal cual, solo limitado para que no se vuelva loco
            factor = max(1.0, min(4.0, phys_factor))

# Ensanchar la imagen (hacerla menos aplastada)
            if HAVE_CV2 and img.ndim == 2:
                h, w = img.shape
                new_w = max(1, int(round(w * factor)))
                interp = cv2.INTER_CUBIC if new_w > w else cv2.INTER_AREA
                img = cv2.resize(img, (new_w, h), interpolation=interp)

# Girar 90° (en el mismo sentido en que antes se veía bien)
            img = np.rot90(img, 1)


# Pasar a QImage/QPixmap
        if img.ndim == 2:
            h, w = img.shape
            bytes_per_line = w
            buffer = img.tobytes()
            qimg = QImage(buffer, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            h, w, c = img.shape
            if c != 3:
                img = np.repeat(img[:, :, None], 3, axis=2)
                h, w, c = img.shape
            bytes_per_line = w * c
            buffer = img.tobytes()
            qimg = QImage(buffer, w, h, bytes_per_line, QImage.Format_RGB888)

        return QPixmap.fromImage(qimg.copy())

# método que informa cuántos cortes tiene cada plano para que la Vista configure los sliders.
    def get_max_slices(self, plane):
        if self.volume is None or self.file_type == "2d" or self.volume.ndim != 3:
            return 0
        plane = plane.lower()
        if plane == "axial":
            return self.volume.shape[2] - 1
        elif plane == "coronal":
            return self.volume.shape[1] - 1
        elif plane == "sagittal":
            return self.volume.shape[0] - 1
        return 0

# método para filtra los 3 cortes (axial, coronal y sagital) al mismo tiempo.
    def apply_filter_to_slices(
        self,
        filter_type: str,
        axial_index: int,
        coronal_index: int,
        sagittal_index: int,
    ):

        if self.volume is None or not HAVE_PYQT:
            return QPixmap(), QPixmap(), QPixmap()

        pix_axial = self.get_slice("axial", axial_index)
        pix_coronal = self.get_slice("coronal", coronal_index)
        pix_sagittal = self.get_slice("sagittal", sagittal_index)

        def pix_to_array(pix: QPixmap) -> Optional[np.ndarray]:
            if pix.isNull():
                return None
            qimg = pix.toImage().convertToFormat(QImage.Format_Grayscale8)
            w = qimg.width()
            h = qimg.height()
            bits = qimg.bits()
            bits.setsize(h * w)
            arr = np.frombuffer(bits, dtype=np.uint8).reshape((h, w))
            return arr

        def array_to_pix(arr: np.ndarray) -> QPixmap:
            h, w = arr.shape
            buffer = arr.tobytes()
            qimg = QImage(buffer, w, h, w, QImage.Format_Grayscale8)
            return QPixmap.fromImage(qimg.copy())

        def filter_array(gray: np.ndarray) -> np.ndarray:
            if not HAVE_CV2:
                return gray
            f = filter_type.lower()
            img = gray.copy()
            if f == "gaussian":
                img = cv2.GaussianBlur(img, (5, 5), 0)
            elif f == "binarizacion":
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            elif f == "canny":
                img = cv2.Canny(img, 100, 200)
            elif f == "equalize":
                img = cv2.equalizeHist(img)
            # "grayscale" o "default": sin cambios fuertes
            return img

        arr_axial = pix_to_array(pix_axial)
        arr_coronal = pix_to_array(pix_coronal)
        arr_sagittal = pix_to_array(pix_sagittal)

        arr_axial_f = filter_array(arr_axial) if arr_axial is not None else None
        arr_coronal_f = filter_array(arr_coronal) if arr_coronal is not None else None
        arr_sagittal_f = filter_array(arr_sagittal) if arr_sagittal is not None else None

        pix_axial_f = array_to_pix(arr_axial_f) if arr_axial_f is not None else pix_axial
        pix_coronal_f = array_to_pix(arr_coronal_f) if arr_coronal_f is not None else pix_coronal
        pix_sagittal_f = array_to_pix(arr_sagittal_f) if arr_sagittal_f is not None else pix_sagittal

        return pix_axial_f, pix_coronal_f, pix_sagittal_f

#Filtro para imágenes 2D (cámara o fotos normales).
    def apply_filter(self, filter_type, image_2d):
        if not HAVE_CV2 or not HAVE_PYQT:
            return QPixmap()

        if image_2d is None:
            if self.volume is None:
                return QPixmap()
            if self.file_type == "2d":
                img = self.volume.astype(np.uint8)
            else:
                if self.volume.ndim == 3:
                    idx = self.volume.shape[2] // 2
                    img = self.volume[:, :, idx]
                else:
                    img = self.volume
        else:
            img = image_2d

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

        f = filter_type.lower()
        if f == "gaussian":
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        elif f == "binarizacion":
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        elif f == "canny":
            gray = cv2.Canny(gray, 100, 200)
        elif f == "equalize":
            gray = cv2.equalizeHist(gray)

        h, w = gray.shape
        qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())

#Convierte una captura de cámara a gris y la guarda en la carpeta temporal del usuario.
    def convert_to_grayscale_and_save(self, frame, session):
        if not HAVE_CV2:
            raise RuntimeError("OpenCV no disponible.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        path = session.get_temp_path(f"perfil_{session.get_username()}.jpg")
        cv2.imwrite(path, gray)
        return path


#La cuarta clase se encarga de procesar señales biomédicas
class SignalProcessor:
    def __init__(self):
        self.signal_data: Dict[int, np.ndarray] = {}
        self.sampling_rate: Optional[float] = None
        self.fft_results: Optional[pd.DataFrame] = None

#método para carga señales desde un .mat en formato consistente para analizarlas
    def load_mat_file(self, path, variable_name):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        mat = loadmat(path)
        keys = [k for k in mat if not k.startswith("__")]
        if not keys:
            raise RuntimeError("No se encontraron variables en el .mat")

        if variable_name and variable_name in mat:
            data = mat[variable_name]
        else:
            data = max((mat[k] for k in keys), key=lambda x: np.size(x))

        arr = np.array(data)
        if arr.ndim == 1:
            self.signal_data = {0: arr.astype(np.float32)}
        elif arr.ndim == 2:
            if arr.shape[0] < arr.shape[1]:
                if arr.shape[0] <= 16:
                    self.signal_data = {i: arr[i, :].astype(np.float32) for i in range(arr.shape[0])}
                else:
                    self.signal_data = {i: arr[:, i].astype(np.float32) for i in range(arr.shape[1])}
            else:
                self.signal_data = {i: arr[i, :].astype(np.float32) for i in range(arr.shape[0])}
        else:
            channels = arr.shape[0]
            self.signal_data = {i: arr[i, :].astype(np.float32) for i in range(channels)}

        if "fs" in mat:
            try:
                self.sampling_rate = float(mat["fs"].squeeze())
            except Exception:
                self.sampling_rate = None
        else:
            sr = mat.get("sampling_rate", None)
            if isinstance(sr, np.ndarray):
                sr = float(np.squeeze(sr))
            self.sampling_rate = sr
        return True

#método que calcula la FFT para todas las señales cargadas
    def compute_fft_all_channels(self) -> pd.DataFrame:
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
            idx = int(np.argmax(yf))
            dominant_freq = float(xf[idx])
            dominant_mag = float(yf[idx])
            rows.append(
                {
                    "channel": int(ch),
                    "dominant_freq": dominant_freq,
                    "dominant_mag": dominant_mag,
                    "freqs": xf.tolist(),
                    "mags": yf.tolist(),
                }
            )
        self.fft_results = pd.DataFrame(rows)
        return self.fft_results

#permite obtener los resultados procesados por el método anterior
    def get_fft_dataframe(self):
        if self.fft_results is None:
            raise RuntimeError("FFT no calculada aún.")
        return self.fft_results

#método que xporta la FFT a un archivo legible para análisis externo
    def save_fft_csv(self, out_path):
        if self.fft_results is None:
            raise RuntimeError("FFT no calculada.")
        df = self.fft_results.copy()
        df["freqs"] = df["freqs"].apply(lambda x: ",".join(map(str, x)))
        df["mags"] = df["mags"].apply(lambda x: ",".join(map(str, x)))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path

#método que crea una gráfica del espectro de un canal específico
    def get_spectrum_plot(self, channel):
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

#Para calcular la variabilidad de la señal y generar histogramas según el tipo de análisis.
    def calculate_std_and_histogram(self, axis: str = "channel") -> Tuple[float, plt.Figure]:
        if not self.signal_data:
            raise RuntimeError("No hay señal cargada.")
        if axis == "channel":
            stds = [float(np.std(sig)) for sig in self.signal_data.values()]
            std_value = float(np.mean(stds))
            data_for_hist = stds
            title = "Histogram of channel std devs"
        elif axis == "global":
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

#La quinta clase se encarga de procesar datos tabulares
class TabularProcessor:
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None

#Este primer método carga un CSV y deja los datos listos para trabajar
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        self.data = pd.read_csv(file_path, **kwargs)
        return self.data

#método que sirve para poblar menús de selección de columnas, combos, filtros, etc
    def get_columns(self):
        if self.data is None:
            return []
        return list(self.data.columns)

#crea una gráfica simple para una columna del CSV
    def get_column_plot(self, column):
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

#este método prepara los datos tabulados para mostrarlos en una tabla de la interfaz
    def get_data_model(self):
        if not HAVE_PYQT:
            raise RuntimeError("PyQt5 no disponible para crear QStandardItemModel.")
        if self.data is None:
            raise RuntimeError("No hay datos cargados.")
        model = QStandardItemModel()
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


#La sexta clase se encarga de registrar la actividad del usuario en una base de datos
class DatabaseLogger:
    def __init__(self, db_type: str = "sqlite", credentials: Optional[Dict[str, Any]] = None):
        # tipo de base de datos: "sqlite" o "mongo"
        self.db_type = db_type.lower()
        # diccionario con la configuración (ruta del .db, URI de mongo, etc.)
        self.credentials = credentials or {}
        self.conn = None
        self.client = None

        if self.db_type == "sqlite":
            # ruta del archivo de base de datos (por defecto pmda_activity.db)
            db_path = self.credentials.get("path", "pmda_activity.db")
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self._ensure_table_sqlite()

        elif self.db_type == "mongo":
            if not HAVE_PYMONGO:
                raise RuntimeError("pymongo no disponible para usar MongoDB.")
            uri = self.credentials.get("uri", "mongodb://localhost:27017")
            dbname = self.credentials.get("db", "pmda")
            coll = self.credentials.get("collection", "activity")
            self.client = MongoClient(uri)
            self.collection = self.client[dbname][coll]

        else:
            raise ValueError("db_type debe ser 'sqlite' o 'mongo'")

    def _ensure_table_sqlite(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                action TEXT,
                timestamp TEXT,
                result_path TEXT
            )
            """
        )
        self.conn.commit()

    def log_activity(self, user: str, action: str, result_path: Optional[str] = None):
        ts = datetime.now().isoformat()
        if self.db_type == "sqlite":
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO activity (username, action, timestamp, result_path) "
                "VALUES (?, ?, ?, ?)",
                (user, action, ts, result_path or ""),
            )
            self.conn.commit()
        else:
            doc = {
                "username": user,
                "action": action,
                "timestamp": ts,
                "result_path": result_path or "",
            }
            self.collection.insert_one(doc)

    def close(self):
        if self.conn:
            self.conn.close()
        if self.client:
            self.client.close()
