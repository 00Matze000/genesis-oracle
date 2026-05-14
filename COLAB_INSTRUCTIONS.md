# Project Genesis: PINN Training in Google Colab

Follow these steps to train your Physics-Informed Neural Network on a GPU/TPU accelerator.

## 1. Setup Environment
Open a new notebook in Google Colab and set the runtime to **GPU** (Runtime -> Change runtime type -> T4 GPU).

## 2. Clone Repository
Run the following cell to pull your code:
```bash
!git clone https://github.com/YOUR_GITHUB_USERNAME/genesis-oracle.git
%cd genesis-oracle
```

## 3. Install dependencies
Run this to install everything needed for the PINN:
```bash
!pip install uv
!uv pip install --system flax optax plotly
```

## 4. Ignition: Start Training
Execute the training script directly from the repository:
```bash
!python train_pinn.py
```


## 5. Download Results
Use the file browser on the left in Colab to navigate to `genesis-oracle/data/` and download `pinn_3d_fabric.html`.
