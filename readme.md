# Phase 1: Open-Source Datasets - Native Speaker Use Case

## Overview
This phase focuses on utilizing open-source datasets to develop sentiment analysis models tailored for native speaker use cases. It includes data preparation, feature engineering, and baseline model evaluations.

---

## Datasets
### **Raw Datasets**
We utilized the following four open-source datasets, each with proper documentation on its source and licensing:

1. **RAVDESS**  
   - **Description:** The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains emotional speech and song recordings by professional actors.  
   - **Source:** [RAVDESS Official Website](https://zenodo.org/record/1188976)  
   - **Licensing:** Creative Commons Attribution 4.0 International License.  

2. **TESS**  
   - **Description:** The Toronto Emotional Speech Set (TESS) includes audio recordings of sentences spoken by two female actors in various emotions.  
   - **Source:** [TESS Dataset Repository](https://tspace.library.utoronto.ca/handle/1807/24487)  
   - **Licensing:** Freely available for non-commercial use.  

3. **SAVEE**  
   - **Description:** The Surrey Audio-Visual Expressed Emotion (SAVEE) dataset contains audio recordings of emotions expressed by male actors.  
   - **Source:** [SAVEE Dataset](http://kahlan.eps.surrey.ac.uk/savee/)  
   - **Licensing:** Use requires permission from the dataset creators.  

4. **CREMA-D**  
   - **Description:** The Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D) features emotional speech recordings in various modalities.  
   - **Source:** [CREMA-D Repository](https://github.com/CheyneyComputerScience/CREMA-D)  
   - **Licensing:** Freely available for research purposes.  

These datasets collectively cover a wide range of emotions labels, providing a diverse and balanced foundation for training. Besides, you could consider datasets like [IEMOCAP](https://sail.usc.edu/iemocap/) as well.

---

## Directory Structure
The following structure organizes all related notebooks into logical steps for data preprocessing, feature extraction, and modeling.

### **`notebooks_data_n_feature_processing/`**
Scripts for **data preprocessing, augmentation, and feature extraction**:
1. **Dataset Preprocessing**
   - `01_dataset_label_processing_23Feb_V2.ipynb`: Label assignment and initial cleaning.
   - `02_dataset_rename_n_train_test_split_23Feb_V2.ipynb`: Renaming and permanent train-test split.
   
2. **Feature Extraction**
   - `03_Feature_extraction_part1_V2.ipynb`: First batch of audio signal features (frequenct/time-domain).
   - `03_Feature_extraction_part2_concat_24Feb.ipynb`: Merging extracted features into a comprehensive set.

3. **Feature Analysis**
   - `04_Feature_correlation_V2_24Feb.ipynb`: Initial correlation analysis to study feature relationships.
   - `04_Feature_correlation_V2_26Feb_3_augment.ipynb`: Augmentation for class balancing and further correlation analysis.
   - `04_Feature_first_pass_RF_importance_24Feb.ipynb`: First-pass feature importance analysis using Random Forest.
   - `04_Feature_Study_Plot_25Feb.ipynb`: Visualizations for feature distribution and relationships.

4. **Advanced Feature Engineering**
   - `05_Data_Augmentation.ipynb`: Strategies for augmenting and balancing data.
   - `06_01_Feature_Eng_Dimension_Reduction_PCA_Prosody.ipynb`: Dimensionality reduction using PCA.
   - `06_Feature_Selection_Prosody_rank.ipynb`: Ranking features by their significance.
   - `06_Feature_Selection_Prosody_Auto_wrapper.ipynb`: Automated feature selection using wrappers.
   - `06_Feature_Selection_grouped_Auto_RFE.ipynb`: Recursive Feature Elimination for optimized feature subsets.    
Note: The final feature combination is decided by Automatic method and manual DOe trial as well.
---

### **`notebooks_model_training_n_evaluation/`**
Scripts for **model training, hyperparameter tuning, and evaluation**:
1. **Experimentation and Training**
   - `12_exp_model_auto.ipynb`: Automated model experimentation pipeline.
   - `12_experiment_models_grid_search.ipynb`: Hyperparameter tuning using grid search for various models.
   - `12_experiment_models_sample.ipynb`: Sampling experiments to improve generalization.

2. **Model-Specific Experiments**
   - `12_experiment_models-20240316-v4-aug-RF.ipynb`: Random Forest with augmented data.
   - `12_experiment_models-20240316-v4-aug-lightgbm.ipynb`: LightGBM experimentation.
   - `12_experiment_models-20240316-v4-aug-SVM.ipynb`: SVM training and evaluation.
   - `13_StackingEnsembleNN_CNN_Attention.ipynb`: Stacked ensemble with CNN and attention mechanisms.

3. **Benchmark with Pretrained Model Evaluations - Wave2Vec** 
   - under `/pretrained_model` folder
   - `11_pretrained_model_evaluation_hubert.ipynb`: Evaluation of HuBERT for audio feature extraction.
   - `11_pretrained_model_evaluation_wave2vec_6cls.ipynb`: Wave2Vec model testing with six sentiment classes.
   - `11_pretrained_model_finetune_wave2vec2_base.ipynb`: Fine-tuning the Wave2Vec2 base model on the dataset.

4. **Benchmark with Pretrained Model Evaluations - Emotion2Vec** 
   - under `/emotion2vec` folder, key observation: **performance drop with augmented signals** 


---

## Notebooks Execution Order
1. **Data Preprocessing**
   - Begin with `01_dataset_label_processing_23Feb_V2.ipynb` to clean and label the data.
   - Follow with `02_dataset_rename_n_train_test_split_23Feb_V2.ipynb` for dataset splitting.

2. **Feature Extraction and Analysis**
   - Execute `03_Feature_extraction_part1_V2.ipynb` and `03_Feature_extraction_part2_concat_24Feb.ipynb`.
   - Perform correlation analysis and feature selection with `04_*` notebooks.

3. **Model Training and Evaluation**
   - Start with pretrained model evaluations in `12_*` notebooks.
   - Proceed to experiment pretrain-finetune models wave2vec & Emotion2Vec

---

## Key Highlights
- **Data Preprocessing:** Ensures high-quality labeled datasets with balanced classes.
- **Feature Engineering:** Extracts and ranks over 128 key features optimized for sentiment analysis.
- **Model Evaluation:** Comprehensive comparison of baseline models and advanced ensembles.
- **Code Reusability:** Modular notebook design for flexible adjustments and scalability.

--- 

For more details on the datasets, preprocessing techniques, and feature engineering workflows, refer to the respective notebooks in the `notebooks_data_n_feature_processing` folder.

## License & Contact

This project is sponsor by NCS, Singapore. This repo is for capstone project code deliverables only.   
**NOT for any commericial usage without grants**  
For any questions or issues, please contact Bianca at e0533381@u.nus.edu .