import os
import glob
import torch
from torch.utils.data import Dataset

class WiFiDataset(Dataset):
    def __init__(self, processed_dir, stage='train'):
        
        self.processed_dir = processed_dir
        self.stage = stage
        
        # Trova tutti i file .pt generati dal preprocessing
        self.file_list = glob.glob(os.path.join(processed_dir, '**', '*.pt'), recursive=True)
        
        # Mappatura delle classi in ID numerici (da 0 a 6 per le 7 attività motorie)
        self.class_to_idx = {f'C{i}': i-1 for i in range(1, 8)} 
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # Caricamento del tensore già pronto (forma originale: 100 x 340)
        doppler_profile = torch.load(file_path)
        
        # Estrazione sicura della classe tramite il nome della cartella genitore
        # Se il file si trova in 'processed_data/train/C1/sample_0.pt', os.path.dirname restituisce '.../C1'
        # os.path.basename su quello restituisce proprio 'C1'
        class_str = os.path.basename(os.path.dirname(file_path)) 
        label = self.class_to_idx[class_str]
        
        # Formattazione finale per PyTorch: (Canali, Altezza/Velocità, Larghezza/Tempo) -> (1, 100, 340)
        # Usiamo unsqueeze(0) per aggiungere la dimensione del canale di ingresso singolo, per poi usare tranquillamente le convolutional 
        tensor_data = doppler_profile.unsqueeze(0)
        
        # Ritorniamo il dato e la label convertita in tensore di tipo Long (per CrossEntropy)
        return tensor_data, torch.tensor(label, dtype=torch.long)