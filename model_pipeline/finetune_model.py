from model_pipeline.data_feeder import SlidingWindowDataset
import torch
from torch.utils.data import DataLoader
import os
from model_pipeline.seq2Point_factory import Seq2PointFactory
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")


def get_worker_init_fn(seed=42):
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return worker_init_fn


class Finetuner:
    def __init__(self, model_state_dir, finetune_csv_dir, dataset, model_save_dir, seed=42):
        set_random_seed(seed)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_state_dir, map_location=self.device)
        self.model_name = checkpoint['model_name']
        self.window_length = checkpoint['window_length']
        self.model_state_dict = checkpoint['model_state_dict']
        self.model = Seq2PointFactory.createModel(self.model_name, self.window_length)
        self.model.load_state_dict(self.model_state_dict)
        self.appliance_name_formatted = checkpoint['appliance']
        self.dataset = dataset 
        self.model_save_dir = model_save_dir
        
        self.freezeLayers()
        
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        beta_1 = 0.9
        beta_2 = 0.999
        learning_rate = 0.001
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, betas=(beta_1, beta_2))
        
        self.batch_size = 1000
        finetune_dataset = SlidingWindowDataset([finetune_csv_dir], self.model.getWindowSize())
        
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        
        self.finetune_loader = DataLoader(
            finetune_dataset, 
            batch_size=1000, 
            shuffle=True,
            generator=self.generator,
            worker_init_fn=get_worker_init_fn(seed)
        )
        self.finetuning_losses = []
        
        self.patience = 5
        self.best_val_loss = float("inf")
        self.min_delta = 1e-8
        self.counter = 0
    
    def freezeLayers(self):
        """
        Freezes all the layers of the model except the fully connected layers
        """
        for name, param in self.model.named_parameters():
            layer_name = name.split('.')[0]
            layer_type = dict(self.model.named_modules()).get(layer_name)
            if not isinstance(layer_type, nn.Linear):
                param.requires_grad = False
    
    def fineTune(self, max_epochs = 50):
        for epoch in range(max_epochs):
            self.model.train()  
            train_loss = 0 
            for inputs,targets in self.finetune_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(-1), targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            
            train_loss /= len(self.finetune_loader)
            print(f"Epoch {epoch+1}/{max_epochs} Loss: {train_loss}")

            if train_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = train_loss
                os.makedirs(self.model_save_dir, exist_ok=True)
                torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'model_name' : self.model_name,
                        'window_length' : self.window_length,
                        'appliance' : self.appliance_name_formatted
                    }, os.path.join(self.model_save_dir,f"{self.appliance_name_formatted}_{self.dataset}_{self.model_name}.pth"))
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            self.finetuning_losses.append(train_loss)
    
    def plotLosses(self, save_location=None):
        """
        Plots the training and validation losses.
        with optional save location
        """
        plt.plot(self.finetuning_losses, label="Finetuning Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        if save_location:
            plt.savefig(save_location)
