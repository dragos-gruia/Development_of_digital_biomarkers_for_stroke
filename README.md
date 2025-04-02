# Development of digital biomarkers for assessing and monitoring cognitive function after stroke

## Executive Summary

This repository contains the pre-processing scripts, and analysis pipelines associated with the paper **"Online monitoring technology for deep phenotyping of cognitive impairment after stroke"** In this study, I developed a digital health assessment technology for deriving clinically-relevant digital biomarkers in patients with cerebrovascular disease. Through a broad spectrum of analyses, I show that the outcomes derived from the technology not only have high validity and reliability, but also outperform existing clinical assessments.

## Statistical techniques applied in the paper

Hierarchical Bayesian modelling, Generalized Linear Models, Factor Analysis, Equivalence Testing, Hypothesis Testing and Non-Parametric Methods

## Research in context

#### **Evidence before this study**

Before conducting this study, the authors reviewed existing evidence related to the assessment and monitoring of cognitive impairment in patients with stroke. The search included databases such as PubMed, Scopus, and Google Scholar, covering studies up to April 2024. Criteria for inclusion included peer-reviewed journal articles and systematic reviews in any language, focusing on digital health technologies and cognitive assessments specific to patient with stroke. Our review of the literature indicated a significant gap in sensitive digital, scalable tools for monitoring cognitive impairment post-stroke that accounts for the variability in cognitive presentations in patients with stroke.

#### **Added value of this study**

This study presents the IC3, a novel online digital platform specifically designed for deep phenotyping of cognitive impairment and neuropsychiatric symptoms in patients with stroke. The IC3 battery stands out for its high test-retest reliability, minimal learning effects, and consistent performance across supervised and unsupervised settings. By leveraging a large normative dataset from over 6000 older adults and employing state-of-the-art Bayesian modelling, the IC3 provides personalised, patient-specific impairment scores with enhanced sensitivity. The IC3 outperforms traditional commonly used clinical assessment tools such as the MoCA in detecting cognitive impairments and shows a stronger correlation with patient-reported functional outcomes, thereby addressing the unmet need for a robust, scalable, and sensitive digital assessment tool.

#### **Implications of all the available evidence**

The combined evidence suggests significant implications for both clinical practice and future research. Clinically, the digital health assessment technology developed here provides a reliable, scalable, and cost-effective method for early detection and monitoring of cognitive impairments, not only in stroke, but also in a plethora of cerebrovascular diseases. Such digital technology aligns with clinical recommendations for comprehensive cognitive screening in stroke management and offers a feasible solution for large-scale cognitive monitoring in diverse healthcare settings. Future research should focus on integrating such a tool into routine clinical practice, and as a scalable outcome measure in clinical trials examining patients with cerebrovascular disease.

<p align="center">

<img src="Figures/task_summary.png"/>

**Figure 1** - Graphical overview of the 22 digital tasks organised by the main cognitive domains tested: memory, language, executive, attention, motor ability, numeracy and praxis. Four optional speech production tasks (naming, repetition, reading, picture description) allow speech to be recorded and manually analysed offline.

</p>

## Repository Structure

├── 1_data_parsing-and-cleaning/ - Scripts for parsing, formatting and cleaning data obtained from the server

├── 2_data_pre_processing/ - Scripts for cleaning the patient and normative cohorts

├── 3_data_analysis/ - Scripts for the analyses discussed in the paper

├── Figures/ - Generated figures and visualizations

├── Docs/ - Additional information on which dependencies to install

├── README.md - This file

└── LICENSE - License information

## Requirements

-   **Programming Languages:** Python 3.10.5 and R 4.3.1

## Installation

1.  **Clone the Repository:**

`bash git clone <https://github.com/dragos-gruia/Development_of_digital_biomarkers_for_stroke>  cd repository-name`

2.  **Install Dependencies:**

Dependencies for the Python and R scripts can be found and installed from the `Docs/` Directory

3.  **Data Download**

The data can be made available on reasonable request and upon institutional regulatory approval. For more information email [dragos-cristian.gruia19\@imperial.ac.uk](mailto:dragos-cristian.gruia19@imperial.ac.uk)

## Running the Analyses

1.  **Data Preparation:** Ensure that all required datasets are placed in a directory called raw_data/

2.  **Execute Data Parsing** Parse data from JSON format to csv and clean it via parsing_functions.py

3.  **Execute Data Pre-processing** Choose which pre-processing pipeline to use based on whether you are working with a patient cohort or with controls

4.  **Execute Analysis Scripts** Run the main analysis scripts in any order by providing as input the data obtained from step 3:

5.  **Output** Generated figures are saved in Figures/directory

## Citation

If you use this repository in your work, please cite the paper as follows:

`Gruia, D.C., Giunchiglia, V., Coghlan, A., Brook, S., Banerjee, S., Kwan, J., Hellyer, P.J., Hampshire, A. and Geranmayeh, F., 2024. Online monitoring technology for deep phenotyping of cognitive impairment after stroke. medRxiv, pp.2024-09. https://doi.org/10.1101/2024.09.06.24313173`

## License

This project is licensed under the Creative Commons License. See the LICENSE file for details.
