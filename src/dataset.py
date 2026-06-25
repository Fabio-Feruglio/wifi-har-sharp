import importlib
import src.dataset
import glob
import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from collections import Counter
import torch
from torch.utils.data import Dataset

class WiFiDataset(Dataset):
    def __init__(self, raw_dir, stage='train'):
        self.raw_dir = raw_dir
        self.stage = stage
        
        # Trova tutti i file .mat
        self.file_list = glob.glob(os.path.join(raw_dir, '**', '*.mat'), recursive=True)
        
        # Mappatura delle classi in ID numerici (es: 'C1' -> 0, 'C2' -> 1...)
        # Nota: adatta questo dizionario in base ai nomi reali delle tue classi!
        self.class_to_idx = {f'C{i}': i-1 for i in range(1, 8)} 
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # 1. Carica i dati MATLAB
        mat_contents = scipy.io.loadmat(file_path)
        csi_data = mat_contents['csi_buff']
        
        # 2. Estrai la classe dal nome del file
        filename = os.path.basename(file_path)
        class_str = filename.replace('.mat', '').split('_')[1] # es: 'C1'
        label = self.class_to_idx[class_str]
        
        # 3. Applica l'estrazione Doppler (la stessa logica provata sopra)
        csi_signal = csi_data[:, 128] # sottoportante centrale
        _, _, spec = spectrogram(csi_signal, fs=1000, nperseg=256, noverlap=192, return_onesided=False)
        spec_shifted = np.fft.fftshift(spec, axes=0)
        
        centro = spec_shifted.shape[0] // 2
        doppler_profile = spec_shifted[centro-50:centro+50, :] # Taglio a 100 bin di velocità
        
        # Se vuoi ridimensionare il tempo esattamente a 340 frame, puoi fare un troncamento o un resize:
        # Per ora lo tagliamo o lo prendiamo così com'è per il test
        doppler_profile = doppler_profile[:, :340] 
        
        # Trasformiamo in Tensore PyTorch (Aggiungendo la dimensione del canale di input = 1)
        # Il modello si aspetta solitamente (Canali, Altezza/Velocità, Larghezza/Tempo)
        tensor_data = torch.tensor(doppler_profile, dtype=torch.float32).unsqueeze(0)
        
        return tensor_data, label