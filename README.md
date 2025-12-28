# Netflix Content Analysis, Text Classification, and Similarity Modelling with PySpark

## Project Overview

This project applies **big data analytics, natural language processing (NLP), and machine learning**
to analyse Netflix content using large-scale textual metadata. The work focuses on two related but
distinct tasks:

1. **Text classification** of Netflix titles using supervised machine learning  
2. **Content-based similarity and recommendation**, identifying similar titles based on text features  

The project was completed as part of the **Big Data Analytics and Data Visualisation** module within
the MSc Data Science programme. It combines **PySpark-based processing**, NLP feature engineering,
machine learning, and **Tableau visualisation** to support scalable analysis and insight generation.

---

## Objectives

- Analyse large-scale Netflix metadata using big data tools  
- Clean and preprocess textual data for machine learning  
- Apply NLP techniques for feature extraction  
- Train and evaluate supervised text classification models  
- Build a content-based similarity system for recommendation  
- Create visual analytics (Tableau dashboards) to communicate insights effectively  

---

## Dataset

- **Source:** Public Netflix titles dataset  
- **Records:** 8,000+ movies and TV shows  
- **Key attributes include:**
  - Title  
  - Description  
  - Genre  
  - Release year  
  - Country  
  - Rating  
  - Duration  

> The dataset is publicly available.  
> If running locally, ensure file paths in the notebooks are updated accordingly.

---

## Tools and Technologies

- **Python**
- **PySpark** (DataFrames, ML pipelines)
- **Pandas / NumPy**
- **Scikit-learn**
- **TF-IDF Vectorisation**
- **Cosine Similarity**
- **Matplotlib / Seaborn**
- **Tableau** for interactive visualisation
- **Jupyter Notebook**

---

## Methodology

### Exploratory Data Analysis and Visualisation

- Analysed Netflix content distribution by type (Movies vs TV Shows)
- Examined trends by release year, country, genre, and rating
- Identified class imbalance and missing values
- Created **Tableau dashboards** to visualise:
  - Content growth over time  
  - Country-level production  
  - Top actors and contributors  
  - Rating and genre distributions  

These visualisations supported both technical analysis and stakeholder-friendly insight communication.

---

### Text Preprocessing and Feature Engineering

- Cleaned and normalised textual fields
- Tokenised text and removed stop words
- Converted text into numerical features using **TF-IDF**
- Prepared scalable ML pipelines using PySpark

---

### Machine Learning Tasks

#### 1. Netflix Text Classification
- Implemented supervised learning models to classify Netflix content
- Compared baseline and improved models
- Evaluated performance using accuracy, precision, recall, and F1-score

#### 2. Content Similarity and Recommendation
- Built a content-based similarity system using **cosine similarity**
- Identified titles most similar to a given movie or TV show
- Demonstrated how text features can support recommendation-style functionality

---

## Evaluation

Models and similarity outputs were assessed using:
- Classification metrics (Accuracy, Precision, Recall, F1-score)
- Qualitative evaluation of similarity rankings
- Visual validation through exploratory analysis and dashboards

---

## Key Findings

- Textual descriptions provide strong signals for both classification and similarity modelling  
- TF-IDF features performed effectively at scale using PySpark  
- Big data tools improved processing efficiency and pipeline structure  
- **Tableau dashboards enhanced interpretability and insight communication**  
- Combining ML with visual analytics strengthened overall analysis quality  

---

## How to Run the Project

1. Open the notebooks using **Google Colaboratory** or **Jupyter Notebook**.

2. Run the notebooks in the following order:
   - `FINAL_NETFLIX_TEXT_CLASSIFICATION.ipynb`  
     (Big data processing, NLP, and supervised text classification)
   - `NETFLIX SIMILARITY.ipynb`  
     (Content-based similarity and recommendation modelling)

3. Ensure dataset paths are correctly configured before execution.

---

## Project Report

The full academic project report is included in this repository and provides a detailed
discussion of the dataset, PySpark pipelines, exploratory analysis, **Tableau dashboards**,
machine learning models, evaluation results, and ethical considerations.

- `Netflix_Big_Data_Text_Classification_Report.pdf`

---

## Repository Structure

Text-classification-with-pyspark
│── FINAL_NETFLIX_TEXT_CLASSIFICATION.ipynb
│── NETFLIX SIMILARITY.ipynb
│── Netflix_Big_Data_Text_Classification_Report.pdf
│── README.md

## Author

**Yvonne Musinguzi**  
MSc Data Science  
Coventry University
