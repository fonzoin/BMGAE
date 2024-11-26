# Bundling-aware Masked Graph AutoEncoder for Bundle Recommendation
This is our implementation for the paper: Bundling-aware Masked Graph AutoEncoder for Bundle Recommendation

## Environment Settings

- torch=1.10.0
- python=3.7
- tqdm=4.66.1
- numpy=1.21.6
- scipy=1.7.3

## Example to run the codes.

1. Run the following command to train BMGAE on Youshu dateset with GPU 0:

    ```
    python3 main.py -g 0 -m BMGAE -d Youshu
    ```

2. After training, you can check the log files in `./logs`

## Parameter Tuning

All the parameters are in `config.yaml`
