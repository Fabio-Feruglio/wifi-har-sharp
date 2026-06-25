import torch
import torch.nn as nn

class SharpClassifier(nn.Module):
    def __init__(self):
        super(SharpClassifier, self).__init__()
        # Un livello di rete neurale finto giusto per testare il codice
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1 * 30 * 100, 2) # Cambia '2' con il tuo numero di classi reale
        print("Modello temporaneo SharpClassifier creato con successo!")

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)