# Incorporating metadata in Traffic Forecasting

#### This project studied several methods on how to better incorporating metadata in model.

## Prerequisites
This project runs on the HPC AI plarform of HKUST(GZ). Before running it, ensure the following requirements are satisfied:
- Python >= 3.9.19
- Other requirements in the `requirements.txt`.

## Data Preparation
- Run the following command for the original data:
```
python generate_data_for_training.py --dataset sd --years 2019
```
- Run the following command for straightforward integration:
```
python generate_data_for_training.py --dataset sd --years 2019 --metadata 1
```
- Run the following command for rest methods:
```
python generate_data_for_training.py --dataset sd --years 2019 --adpadj 1
```
## Method Running

1. Straightforward integration
- Change the default value of `input_dim` parameter to 5 under the path `src/utils/args.py`.
- Uncomment the `# from src.models.gwnet_original import GWNET` line in file `experiments/gwnet/main.py`.
2. Additional matrix integration
- Uncomment the `# from src.models.gwnet_meta_matrix import GWNET` line in file `experiments/gwnet/main.py`.
3. meta TCN
- Uncomment the `# from src.models.gwnet_meta_tcn import GWNET` line in file `experiments/gwnet/main.py`.
4. meta GAT
- Uncomment the `# from src.models.gwnet_meta_gat import GWNET` line in file `experiments/gwnet/main.py`.
5. Final running
- Execute the bash command in the terminal:
```
bash experiments/gwnet/run.sh
```
