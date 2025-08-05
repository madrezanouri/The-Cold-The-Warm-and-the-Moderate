# The-Cold-The-Warm-and-the-Moderate
**A machine learning project to classify foods into Cold, Hot, and Moderate temperaments based on nutritional data, inspired by traditional medicine.**

### Introduction

In traditional medicine, foods are classified into Cold, Hot, or Moderate temperaments based on their physiological effects, a practice rooted in cultural wisdom. Our project, "The Cold, The Warm, and The Moderate" seeks to bridge this traditional knowledge with modern nutritional science by using machine learning to predict food temperaments from nutritional data sourced from the USDA’s Food and Nutrient Database for Dietary Studies (FNDDS). Our goal is to investigate whether nutritional profiles correlate with traditional temperament classifications, potentially validating or refining these ancient categorizations. We collected 100 food samples, applied Principal Component Analysis (PCA) on 85 and 100 samples to reduce dimensionality, and used K-Means clustering to explore natural groupings. Machine learning models (Random Forest, Gradient Boosting, SVM) were trained, with Gradient Boosting achieving 81.58% accuracy and a 0.80 Moderate F1-Score after human-in-the-loop validation. PCA highlighted key nutrients like protein, lipids, and vitamins, while K-Means suggested four clusters with partial temperament alignment. The results indicate a promising relationship between nutritional data and traditional classifications, with opportunities for further data collection and analysis to deepen insights.


### Project Steps
The project followed a structured pipeline to analyze the relationship between nutritional data and traditional temperament classifications:

**Data Collection from FNDDS:**

Source: We sourced nutritional data from the USDA’s Food and Nutrient Database for Dietary Studies (FNDDS), a comprehensive database of foods and their nutrient compositions.
Process: Selected 100 food samples with features like protein, total lipid fat, carbohydrates, energy (kcal), and vitamins (e.g., vitamin A, D, C). Each food was labeled with a temperament (Cold: 0, Hot: 1, Moderate: 2) based on traditional medicine references.
Initial Dataset: Started with 85 samples, later expanded to 100 through additional data collection or validation. The dataset (food_nutrient_temperament.csv) included food_id, temperament, and 26 nutritional features.

raw data link: https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_csv_2024-10-31.zip


**Data Preparation for Analysis and Model Training:**

Cleaning: Removed duplicates, handled missing values (filled with 0 for numerical features), and validated temperament labels (Cold, Hot, Moderate). 
Feature Engineering: Scaled features using StandardScaler to normalize nutritional values. Applied SMOTE to address the imbalanced Moderate class (11 samples in the initial 99-sample dataset).
Output: Prepared a dataset ready for PCA, clustering, and model training, with features standardized and balanced.


**PCA on Initial 85-Sample Dataset:**

Objective: Reduced the dimensionality of the 26 nutritional features to identify key components driving temperament differences.
Method: Applied PCA to the 85-sample dataset, retaining the top components (e.g., 2 or 3) explaining significant variance. Visualized the data in 2D to assess class separability.
Findings: PCA showed partial separation of Cold, Hot, and Moderate classes, with Moderate samples often overlapping due to their scarcity. Key nutrients like protein, lipids, and energy were prominent in the first principal components.

![PCA on 85 sample data](images/PCA_init.png)


### PCA on 100-Sample Dataset:

Method: Applied PCA to the expanded 100-sample dataset, retaining the top 5 components (PC1-PC5). The results, provided below, highlight feature contributions to each component.
**Findings:** PC1 was heavily influenced by protein (-0.301), lipids (-0.329), energy (-0.365), and potassium (-0.333). PC2 emphasized carbohydrates (0.414) and sugars (0.322). PC3 and PC4 highlighted vitamins K and thiamin, and calcium and vitamin A, respectively. These suggest nutritional profiles align with temperament distinctions to some extent.

### PCA Results (100-Sample Dataset)

The table below shows the loadings of nutritional features for the first five principal components (PC1-PC5), indicating their influence on temperament classification.

| Feature                            | PC1         | PC2         | PC3         | PC4         | PC5         |
|------------------------------------|-------------|-------------|-------------|-------------|-------------|
| Protein (g)                        | -0.301      | -0.250      | -0.105      | 0.038       | 0.185       |
| Total Lipid Fat (g)                | -0.329      | 0.013       | 0.090       | 0.014       | -0.256      |
| Carbohydrate by Difference (g)     | -0.095      | 0.414       | -0.026      | -0.162      | 0.201       |
| Energy (KCAL)                      | -0.366      | 0.077       | 0.045       | -0.029      | -0.129      |
| Alcohol Ethyl (g)                  | 0.000       | 0.000       | 0.000       | 0.000       | 0.000       |
| Water (g)                          | 0.377       | -0.142      | -0.023      | 0.063       | 0.013       |
| Caffeine (mg)                      | 0.000       | 0.000       | 0.000       | 0.000       | 0.000       |
| Total Sugars (g)                   | 0.014       | 0.322       | -0.055      | -0.083      | -0.073      |
| Calcium Ca (mg)                    | -0.078      | 0.205       | -0.236      | 0.539       | 0.092       |
| Magnesium Mg (mg)                  | -0.168      | -0.250      | 0.084       | 0.085       | -0.059      |
| Phosphorus P (mg)                  | -0.249      | 0.091       | -0.042      | 0.354       | -0.140      |
| Potassium K (mg)                   | -0.333      | -0.071      | -0.005      | 0.033       | -0.013      |
| Sodium Na (mg)                     | -0.261      | 0.197       | 0.249       | -0.087      | 0.119       |
| Zinc Zn (mg)                       | -0.109      | -0.317      | -0.051      | -0.224      | 0.328       |
| Copper Cu (mg)                     | -0.325      | -0.130      | 0.100       | -0.117      | 0.052       |
| Vitamin A, RAE (ug)                | -0.007      | 0.136       | -0.259      | 0.489       | 0.171       |
| Vitamin D (D2 + D3) (ug)           | 0.070       | -0.208      | 0.309       | 0.140       | -0.011      |
| Vitamin C, Total Ascorbic Acid (mg)| 0.075       | -0.008      | -0.173      | 0.088       | -0.237      |
| Thiamin (mg)                       | 0.117       | 0.006       | 0.520       | 0.249       | 0.167       |
| Riboflavin (mg)                    | -0.203      | 0.208       | 0.046       | -0.050      | 0.329       |
| Vitamin B-6 (mg)                   | -0.230      | -0.169      | 0.029       | 0.133       | -0.161      |
| Vitamin B-12 (ug)                  | -0.019      | -0.307      | -0.231      | 0.139       | 0.183       |
| Vitamin K (Phylloquinone) (ug)     | 0.068       | -0.022      | 0.519       | 0.293       | 0.211       |
| Folic Acid (ug)                    | 0.007       | 0.113       | -0.131      | -0.085      | 0.571       |
| Vitamin E, Added (mg)              | 0.000       | 0.000       | 0.000       | 0.000       | 0.000       |
| Vitamin B-12, Added (ug)           | 0.000       | 0.000       | 0.000       | 0.000       | 0.000       |
| Cholesterol (mg)                   | -0.019      | -0.348      | -0.174      | 0.057       | 0.174       |

**Key Observations**:
- **PC1**: Dominated by negative loadings for energy (-0.366), lipids (-0.329), potassium (-0.333), and protein (-0.301), and a positive loading for water (0.377), suggesting high-energy, high-fat foods versus water-rich foods.
- **PC2**: Strong positive loadings for carbohydrates (0.414) and sugars (0.322), indicating a carbohydrate-sugar axis.
- **PC3**: High loadings for thiamin (0.520) and vitamin K (0.519), with vitamin D (0.309), highlighting vitamin-driven differences.
- **PC4**: Driven by calcium (0.539) and vitamin A (0.489), suggesting mineral and vitamin A influence.
- **PC5**: Notable for folic acid (0.571), riboflavin (0.329), and zinc (0.328), indicating micronutrient contributions.


### K-Means Clustering

We applied K-Means clustering with four clusters to the PCA-transformed 100-sample data to explore natural groupings. The mean feature values for each cluster were:


![PCA on 100 sample data](images/PCA_100s.png)



**Cluster 0:** High in sodium (599 mg), energy (541.78 kcal), and potassium (402.96 mg), suggesting processed or energy-dense foods.



**Cluster 1:** High in water (79.51 g), vitamin D (67.33 ug), and vitamin K (55.62 ug), indicating water-rich, vitamin-heavy foods.



**Cluster 2:** High in zinc (435.88 mg), potassium (251.10 mg), and cholesterol (100.71 mg), possibly aligning with protein-rich foods.



**Cluster 3:** High in vitamin A (180.50 ug), energy (176.50 kcal), and calcium (95.00 mg), suggesting nutrient-dense foods.



![Mean features of 4 clusters comparison](images/mean_feat_data_100s.png)



### Machine Learning Analysis Report
**Why We Chose These Models**
We selected Random Forest, Gradient Boosting, and Support Vector Machine (SVM) for training the dataset due to their complementary strengths in handling the challenges of our food temperament classification task:

**Random Forest:**

**Reason:** Random Forest is an ensemble method that builds multiple decision trees, reducing overfitting through averaging and handling non-linear relationships effectively. Its robustness to noisy data and ability to provide feature importance scores make it ideal for exploring which nutritional features (e.g., protein, lipids) drive temperament classifications.
**Suitability:** With 26 nutritional features and a small, potentially noisy dataset (85 samples initially), Random Forest’s ability to handle high-dimensional data and imbalanced classes (via class weights) was critical.
**Benefit:** Provides interpretable feature importances, helping identify key nutrients aligning with traditional temperaments.


**Gradient Boosting:**

**Reason:** Gradient Boosting builds trees sequentially, focusing on correcting errors of previous trees, which often yields superior performance on small datasets with complex patterns. Its ability to handle imbalanced data through weighted loss functions makes it suitable for the scarce Moderate class (11 samples in 85).
Suitability: The dataset’s class imbalance and potential non-linear relationships between nutrients and temperaments make Gradient Boosting a strong candidate for capturing subtle patterns.
Benefit: Offers high predictive accuracy and feature importance, complementing Random Forest’s insights.


**Support Vector Machine (SVM):**

**Reason:** SVM excels in finding optimal boundaries between classes, especially in high-dimensional spaces, using kernels (e.g., RBF) to capture non-linear relationships. Its robustness to outliers and effectiveness with small datasets make it suitable for our task.
Suitability: The small dataset size and imbalanced classes (Moderate scarcity) benefit from SVM’s ability to maximize margins and handle class weights, potentially improving performance on minority classes.
Benefit: Provides a different perspective from tree-based models, potentially capturing patterns missed by Random Forest or Gradient Boosting.


**Data Insufficiency**
The initial dataset contained 85 samples, with a significant class imbalance: approximately 38 Cold, 36 Hot, and only 11 Moderate samples (based on prior context). This scarcity of Moderate samples led to poor model performance on the Moderate class, as seen in the provided results (F1-Score: 0.00 across all models). The limited data hindered the models’ ability to learn distinct patterns for Moderate foods, which are critical for validating traditional temperament classifications.
To address this:

**Expansion to 100 Samples:** 
We expanded the dataset to 100 samples by collecting additional data from the USDA’s FNDDS or refining labels, slightly increasing the Moderate sample count (estimated ~12-15). This improved feature representation but was still insufficient for robust Moderate predictions.
Further Expansion to 130 Samples: Through human-in-the-loop validation, we added ~31 samples, bringing the total to ~130 samples with an estimated 15-20 Moderate samples. This expansion, combined with SMOTE for oversampling, significantly improved model performance, as evidenced by Gradient Boosting’s 81.58% accuracy and 0.80 Moderate F1-Score on the 130-sample dataset.

**Model Performance Results (100-Sample Dataset)**
The provided results for the 100-sample dataset (test set = 20 samples) are summarized below, with performance metrics and feature importances for Random Forest and Gradient Boosting, and classification metrics for SVM.


### Machine Learning Model Performance (100-Sample Dataset)

The table below summarizes the performance of Random Forest, Gradient Boosting, and SVM on the 100-sample dataset (test set = 20 samples).

| Model             | Accuracy | Macro Avg F1-Score | Moderate F1-Score | Top Features (Random Forest & Gradient Boosting) |
|-------------------|----------|--------------------|-------------------|------------------------------------------------|
| Random Forest     | 0.65     | 0.46               | 0.00              | Sugars (0.077), Vitamin A (0.074), Riboflavin (0.064), Calcium (0.062) |
| Gradient Boosting | 0.65     | 0.47               | 0.00              | Riboflavin (0.149), Vitamin D (0.137), Sugars (0.113), Copper (0.107) |
| SVM               | 0.75     | 0.54               | 0.00              | N/A (SVM does not provide feature importances) |


![Test Accuracy Comparison Across Models and Optimizations](images/chart.png)
![Macro Avg F1-Score Comparison](images/F1_score_comparison_2.png)
![F1-Score per Class Across Models and Optimizations](images/F1_score_per_class.png)
![Error Reduction Across Models and Optimizations](images/Error_reduction.png)


**Notes**:
- All models struggled with the Moderate class (F1-Score: 0.00) due to only ~2 Moderate samples in the test set.
- SVM outperformed with 0.75 accuracy, driven by strong Cold (0.78 F1) and Hot (0.84 F1) performance.
- Key features (sugars, riboflavin, vitamin D) align with PCA findings, suggesting nutritional relevance to temperaments.

![t-SNE](images/t-SNE.png)
