import os
import glob
import scipy.io
import torch
import numpy as np
from scipy.signal import spectrogram
from tqdm import tqdm

def process_and_save(raw_dir, output_dir):
    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    file_list = glob.glob(os.path.join(raw_dir, '**', '*.mat'), recursive=True)
    
    print(f"Trovati {len(file_list)} file. Inizio elaborazione...")
    
    for file_path in tqdm(file_list):
        # 1. Carica
        mat_contents = scipy.io.loadmat(file_path)
        csi_data = mat_contents['csi_buff']
        
        # 2. Estrazione Doppler (la tua logica)
        csi_signal = csi_data[:, 128] 
        _, _, spec = spectrogram(csi_signal, fs=1000, nperseg=256, noverlap=192, return_onesided=False)
        spec_shifted = np.fft.fftshift(spec, axes=0)
        
        centro = spec_shifted.shape[0] // 2
        doppler_profile = spec_shifted[centro-50:centro+50, :340]
        
        # Padding se necessario
        if doppler_profile.shape[1] < 340:
            pad_width = 340 - doppler_profile.shape[1]
            doppler_profile = np.pad(doppler_profile, ((0,0), (0, pad_width)), mode='constant')
            
        # 3. Conversione e salvataggio
        tensor_data = torch.tensor(doppler_profile, dtype=torch.float32)
        
        # Mantieni la struttura delle cartelle originale
        relative_path = os.path.relpath(file_path, raw_dir)
        save_path = os.path.join(output_dir, relative_path.replace('.mat', '.pt'))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save(tensor_data, save_path)

if __name__ == "__main__":
    process_and_save('raw_data', 'processed_data')
