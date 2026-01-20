# Bundling-aware Masked Graph AutoEncoder for Bundle Recommendation
This is the official implementation of our ICASSP 2026 paper: Bundling-aware Masked Graph AutoEncoder for Bundle Recommendation

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

## B-I Correlation Heatmap Analysis

<img width="6455" height="1915" alt="heatmap1" src="https://github.com/user-attachments/assets/ce1c5b72-84c8-423a-a36f-1de53715ce7f" />

To validate the model’s effectiveness in capturing bundling strategies, we randomly select representations of three bundles and twelve items they contain within the iFashion dataset for analysis. Specifically, Bundle 2496 comprises Item 26556, 36542, and 39498; Bundle 23298 includes Item 19154, 21967, 27842, 34339, and 40722; and Bundle 18117 contains Item 20694, 35530, 36085, and 40334. By calculating the cosine distance for each B-I pair, we obtain the correlation scores between bundle and item representations. These correlation scores are visualized as a heatmap, as shown in the figure. The color intensity of each cell in the heatmap intuitively demonstrates the correlation between different B-I pairs.

Analysis of the figure reveals the following insights: 1) Each bundle exhibits a high correlation with the items it contains, while the correlation with items it does not contain is extremely low. This indicates that BMGAE effectively models the deep dependencies between bundles and items, capturing the implicit bundling-strategy features. 2) There is a significant difference in correlation scores between intra-bundle items and extra-bundle items, further validating that BMGAE significantly enhances the distinction of representations, effectively capturing the bundling-strategy features.
