# oncology-reliability-test
An AI-Assisted Oncology Research Intelligence System for Extracting Clinical Study Information and Predicting Research Reliability

## Data Extraction
 extractor.py - Contains the `OncologyExtractor` class which uses Google's Gemini model to extract structured information (like patient population, study design, results) from oncology research PDF papers. It handles PDF text extraction and API interaction.
 models.py - Defines Pydantic models (`PatientPopulation`, `StudyDesign`, `OncologyStudy`) that structure the data extracted from the research papers, ensuring type safety and consistent data format.
 features.py - Performs feature engineering on the extracted JSON data. It loads the data, performs label encoding on the reliability score, creates structured features (like study type indicators, sample size tiers), and prepares the data for modeling.

 ## Dataset
 final_features.csv - A CSV dataset containing the processed features and extracted information for each research paper, including metadata, study design details, results, and the target variable (reliability label).

 ## Feature Extraction
 feature_engineering.ipynb - A Jupyter notebook that prototypes and visualizes the feature engineering process. It loads data, performs transformations, and generates plots to understand the feature distribution.

 ## Modeling and Results
 ReliabilityModeling.ipynb - A Jupyter notebook focused on building and evaluating machine learning models to predict the reliability of the research papers. It includes data splitting, model training, evaluation metrics, and result visualization.