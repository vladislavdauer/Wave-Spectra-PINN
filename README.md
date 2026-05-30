# Wave Spectra PINN

## Структура проекта

```text
Wave-Spectra-PINN/                        
├── train.py            
├── train_notebook.ipynb    
├── requirements.txt       
├── output/               
│    ├── checkpoints/       
│    │   ├── best_model.pt
│    │   └── final_model.pt
│    ├── logs/               
│    │   ├── history.csv
│    │   └── metrics.txt
│    ├── plots/             
│    │   ├── bands_pinn_brentq.png
│    │   ├── error.png
│    │   └── loss.png
│    └── data.npz  
├── sanity_output/  
│    └─── ...
├── README.md   
└── .gitignore
```

## Jupyter Notebook

```bash

jupyter notebook train_notebook.ipynb
```

## Установка окружения на Linux

```bash

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

## Sanity check

```bash

python train.py \
    --output-dir sanity_output \
    --sanity
```

## Полноценное обучение


```bash

python train.py \
    --output-dir output \
    --bands 4 \
    --points-per-segment 100 \
    --root-scan-points 6000 \
    --lattice-terms 8 \
    --mu 10.0 \
    --epochs 10000 \
    --batch-size 64 \
    --hidden-dim 128 \
    --hidden-layers 5 \
    --learning-rate 8e-4 \
    --lbfgs-steps 30
```

## Воспроизводимость

```python
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
