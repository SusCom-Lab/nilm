from model_pipeline.data_feeder import SlidingWindowDataset
import torch
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
from model_pipeline.seq2Point_factory import Seq2PointFactory
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


class Trainer:
    def __init__(self, model_name, train_csv_dirs, appliance, dataset, model_save_dir, window_length=599, val_ratio=0.2, result_dir=None, seed=42):
        """
        Trainer class for training Seq2Point models.
        model_name (str): Name of the model to train.
        train_csv_dirs (list): List of file paths to the training CSVs.
        appliance (str): Name of the appliance to train the model for.
        dataset (str): Name of the dataset.
        model_save_dir (str): Directory to save the trained model.
        window_length (int): Length of the input window.
        val_ratio (float): Ratio of validation set split from training data (default 0.2 = 20%).
        result_dir (str): Directory to save results (default: result/{dataset}/).
        seed (int): Random seed for reproducibility (default: 42).
        """
        set_random_seed(seed)
        
        self.model_name = model_name
        self.model = Seq2PointFactory.createModel(self.model_name, window_length)

        self.appliance = appliance
        self.appliance_name_formatted = self.appliance.replace(" ", "_")
        self.dataset = dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.model_save_dir = model_save_dir
        
        if result_dir is None:
            self.result_dir = os.path.join("result", dataset.lower(), self.appliance_name_formatted)
        else:
            self.result_dir = os.path.join(result_dir, dataset.lower(), self.appliance_name_formatted)
        os.makedirs(self.result_dir, exist_ok=True)

        self.criterion = nn.MSELoss()
        beta_1 = 0.9
        beta_2 = 0.999
        learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=0.1)

        self.batch_size = 256
        train_split_ratio = 1 - val_ratio
        train_dataset = SlidingWindowDataset(train_csv_dirs, self.model.getWindowSize(), split_ratio=train_split_ratio, split_mode='train')
        validation_dataset = SlidingWindowDataset(train_csv_dirs, self.model.getWindowSize(), split_ratio=train_split_ratio, split_mode='val')
        
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            generator=self.generator,
            worker_init_fn=get_worker_init_fn(seed)
        )
        self.validation_loader = DataLoader(
            validation_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            worker_init_fn=get_worker_init_fn(seed)
        )

        self.patience = 8
        self.best_val_loss = float("inf")
        self.min_delta = 1e-4
        self.counter = 0
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, threshold=self.min_delta)

        self.train_losses = []
        self.val_losses = []

    def trainModel(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(-1), targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in self.validation_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.squeeze(-1), targets)
                    val_loss += loss.item()

            train_loss /= len(self.train_loader)
            val_loss /= len(self.validation_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss - self.min_delta:
                print(f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving model...")
                self.best_val_loss = val_loss
                if not os.path.exists(self.model_save_dir):
                    os.makedirs(self.model_save_dir)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_name' : self.model_name,
                    'window_length' : self.model.getWindowSize(),
                    'appliance' : self.appliance_name_formatted
                }, os.path.join(self.model_save_dir, f"{self.appliance}_{self.dataset}_{self.model_name}.pth"))
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered. No improvement in validation loss for {self.patience} epochs.")
                    break

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

    def plotLosses(self):
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Training and Validation Loss for {self.appliance_name_formatted} on {self.dataset} using {self.model_name}")
        plot_filename = f'{self.appliance_name_formatted}_{self.dataset}_{self.model_name}_loss.png'
        plot_path = os.path.join(self.result_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss plot saved to {plot_path}")
