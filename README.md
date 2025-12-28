# Text-classification-with-pyspark
# Netflix Content Text Classification and Analysis

## Project Overview

This project applies **big data analytics, natural language processing (NLP), and machine learning** techniques to analyse and classify Netflix content using textual metadata. The aim is to explore patterns within Netflix titles and evaluate how text-based features can support automated content classification and insight generation for media platforms.

The project was completed as part of the **Big Data Analytics and Data Visualisation** module within the MSc Data Science programme. It combines large-scale data processing, exploratory data analysis, interactive visualisation, and supervised machine learning.

---

## Objectives

- Perform exploratory data analysis on a large Netflix metadata dataset  
- Process and analyse data using big data tools  
- Clean and preprocess textual data for machine learning  
- Apply NLP techniques for feature extraction  
- Train and evaluate machine learning models for content classification  
- Create interactive visualisations to communicate insights effectively  

---

## Dataset

- **Source:** Public Netflix titles dataset  
- **Records:** Over 8,000 Netflix movies and TV shows  
- **Key features include:**
  - Title  
  - Description  
  - Genre  
  - Release year  
  - Country  
  - Rating  
  - Duration  

> The dataset is not included in this repository.  
> Please download it from the original source and update file paths in the notebook accordingly.

---

## Tools and Technologies

- **Python**
- **PySpark** for large-scale data processing
- **Pandas / NumPy**
- **Scikit-learn**
- **TF-IDF Vectorisation**
- **Matplotlib / Seaborn**
- **Tableau** for interactive data visualisation
- **Jupyter Notebook**

---

## Methodology

### Exploratory Data Analysis

- Analysed content distribution by type (Movies vs TV Shows)
- Examined trends by release year, country, rating, and genre
- Identified missing values and inconsistencies in metadata
- Created interactive dashboards using **Tableau** to visualise key trends
- Used Python-based visualisations to support and validate insights

### Data Preprocessing

- Removed duplicates and handled missing values
- Cleaned and normalised textual fields
- Tokenised text and removed stop words
- Prepared text data for machine learning pipelines

### Feature Engineering

- Converted textual descriptions into numerical representations
- Applied **TF-IDF vectorisation** to capture important keywords
- Selected relevant features for classification tasks

### Machine Learning Models

- Implemented supervised classification models
- Compared baseline and improved models
- Evaluated models using standard performance metrics

---

## Evaluation

Models were evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  

Performance comparisons were used to assess the effectiveness of text-based
classification approaches on Netflix content metadata.

---

## Key Findings

- Textual descriptions provide strong signals for Netflix content classification  
- TF-IDF features performed effectively for modelling content categories  
- Big data tools improved scalability and processing efficiency  
- **Tableau dashboards enabled clear communication of trends and insights**  
- Combining visual analytics with machine learning strengthened overall analysis  

---

## Data Visualisation

To support exploratory analysis and insight generation, **interactive dashboards
were created using Tableau**. These visualisations examined content type
distribution, release patterns over time, genre popularity, and country-level
production, enabling insights to be communicated clearly to non-technical
stakeholders.

---

## How to Run the Project

1. Open the notebook using **Google Colaboratory** or **Jupyter Notebook**.
2. Download the Netflix dataset and update the file paths in the notebook.
3. Run all cells in:
   - `FINAL_NETFLIX_TEXT_CLASSIFICATION.ipynb`

---

## Project Report

The full project report is included in this repository and provides a detailed
academic discussion of the dataset, exploratory analysis, **Tableau dashboards**,
text preprocessing, feature engineering, machine learning models, evaluation
results, and ethical considerations.

- `Netflix_Content_Text_Classification_Report.pdf`

---

## Repository Structure
Netflix-Content-Classification
│── FINAL_NETFLIX_TEXT_CLASSIFICATION.ipynb
│── Netflix_Content_Text_Classification_Report.pdf
│── README.md

## Author

**Yvonne Musinguzi**  
MSc Data Science  
Coventry University
