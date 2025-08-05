# The-Cold-The-Warm-and-the-Moderate
**A machine learning project to classify foods into Cold, Hot, and Moderate temperaments based on nutritional data, inspired by traditional medicine.**

**Introduction**

In traditional medicine, foods are classified into Cold, Hot, or Moderate temperaments based on their physiological effects, a practice rooted in cultural wisdom. Our project, "The Cold, The Warm, and The Moderate" seeks to bridge this traditional knowledge with modern nutritional science by using machine learning to predict food temperaments from nutritional data sourced from the USDA’s Food and Nutrient Database for Dietary Studies (FNDDS). Our goal is to investigate whether nutritional profiles correlate with traditional temperament classifications, potentially validating or refining these ancient categorizations. We collected 100 food samples, applied Principal Component Analysis (PCA) on 85 and 100 samples to reduce dimensionality, and used K-Means clustering to explore natural groupings. Machine learning models (Random Forest, Gradient Boosting, SVM) were trained, with Gradient Boosting achieving 81.58% accuracy and a 0.80 Moderate F1-Score after human-in-the-loop validation. PCA highlighted key nutrients like protein, lipids, and vitamins, while K-Means suggested four clusters with partial temperament alignment. The results indicate a promising relationship between nutritional data and traditional classifications, with opportunities for further data collection and analysis to deepen insights.


**Step 2: Project Steps**
The project followed a structured pipeline to analyze the relationship between nutritional data and traditional temperament classifications:

**Data Collection from FNDDS:**

Source: We sourced nutritional data from the USDA’s Food and Nutrient Database for Dietary Studies (FNDDS), a comprehensive database of foods and their nutrient compositions.
Process: Selected 100 food samples with features like protein, total lipid fat, carbohydrates, energy (kcal), and vitamins (e.g., vitamin A, D, C). Each food was labeled with a temperament (Cold, Hot, Moderate) based on traditional medicine references.
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
