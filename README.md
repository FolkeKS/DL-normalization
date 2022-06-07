# DL-normalization

Deep learning methods for estimating normalization coefficients


## Set up conda environment:

From project directory, run:

<code> conda env create --file environment.yml </code> 

<code> conda activate DL-normalization </code> 

<code> pip install -e . </code> 

## Set up wandb for experiment tracking:

<ol>
  <li>Sign up at https://wandb.ai/site and log in</li>
  <li>Find your API-key at https://wandb.ai/authorize </li>
  <li>In your conda environment, run <code> wandb login </code> and provide API-key </li>
</ol> 

## Train model on demonstration data :

<code> python scripts/trainer.py fit --config  configs/demo.yaml </code> 
