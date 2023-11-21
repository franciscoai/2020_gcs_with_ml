import configparser
import ast
import torch
from pathlib import Path


class Configuration:
    def __init__(self, ini_path):
        self.ini_path = ini_path
        self.config = configparser.ConfigParser()
        self.config.read(self.ini_path)
    
    def read_values(self):
        # General Configuration
        self.train_dir = self.config.get("General Configuration", 'training_data_path')
        self.opath = self.config.get("General Configuration", 'output_path')
        self.binary_mask = self.config.getboolean("General Configuration", 'binary_mask')
        self.batch_size = self.config.getint("General Configuration", 'batch_size')
        self.batch_limit = self.config.getint("General Configuration", 'batch_limit')
        self.rnd_seed = self.config.getint("General Configuration", 'rnd_seed')
        self.img_size = self.config["General Configuration"]['img_size'].split(',')
        self.img_size = [float(i) for i in self.img_size]
        self.device = self.config.getint("General Configuration", 'device')
        self.inference_mode = self.config.getboolean("General Configuration", 'inference_mode')
        self.save_model = self.config.getboolean("General Configuration", 'save_model')
        self.load_model = self.config.getboolean("General Configuration", 'load_model')

        # Training Configuration
        self.epochs = self.config.getint("Training Configuration", 'epochs')
        self.train_index_size = self.config.getint("Training Configuration", 'train_index_size')
        self.lr = self.config["Training Configuration"]['lr'].split(',')
        self.lr = [float(i) for i in self.lr]
        self.par_rng = self.config["Training Configuration"]['par_rng'].split(',')
        self.par_rng = [ast.literal_eval(i) for i in self.par_rng]
        self.par_loss_weight = torch.tensor(self.calculate_weights(self.par_rng[:6]))

    def calculate_weights(self, ranges):
        return [10.0 / (max_val - min_val) for min_val, max_val in ranges]