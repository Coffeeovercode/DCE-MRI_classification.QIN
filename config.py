import torch

# --- Project Paths ---
DATA_PATH = "data/"
OUTPUT_PATH = "outputs/"

# --- Dataset Parameters ---
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False # Set to True to load a pre-trained model for continued training