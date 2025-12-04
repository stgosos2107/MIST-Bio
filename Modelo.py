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