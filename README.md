# CVPRW复现
- Result
```txt
Epoch 250
    default_norm
        Best Epoch 221, Test Acc 75.92%
    recovered
        Best Epoch 198, Test Acc 79.39%
    hidden
        Best Epoch 249, Test Acc 83.08%
Epoch 140
    default_norm
        Best Epoch 134, Test Acc 69.63%
    recovered
        Best Epoch 137, Test Acc 75.70%
    hidden
        Best Epoch 135, Test Acc 74.19%
    
```
- Baseline

    `sh ./scripts/default_norm.sh`
- Generate clean data from auto-encoder
    
    `sh ./scripts/train_ae_trail.sh`
    `sh ./scripts/use_ae_trail.sh`

- Final result

    `sh ./scripts/hidden.sh` 

    and 

    `sh ./scripts/recovered.sh` 
- Visualization

    Raw data: `python3 feeder.py --local --vid a01_s01_e00_v2_skeleton --no_norm`

    Normed data: `python3 feeder.py --local --vid a01_s01_e00_v2_skeleton --norm `

    Final clean data: `python3 feeder.py --local --vid a01_s01_e00_v2_skeleton --modality _recovered `