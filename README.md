# DL-normalization
Deep learning methods for estimating normalization coefficients


conda env create --file environment.yml

conda activate DL-normalization

pip install -e .

python scripts/trainer.py fit --config  configs/demo.yaml
