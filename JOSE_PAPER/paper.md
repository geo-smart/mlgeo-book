---
title: 'Machine Learning in Geosciences'
tags:
  - Python
  - PyTorch
  - Machine Learning
  - Jypyter Book
  - Jupyter notebook
  - Geosciences
  - Deep Learning
  - Data Science
  - Graduate Curriculum
  - Undergraduate Curriculum
authors:
  - name: Marine Denolle
    orcid: 0000-0002-1610-2250
    affiliation: "1"
    corresponding: true
  - name: Nicoleta Cristea
    affiliation: "1"
    orcid: 0000-0002-9091-0280
  - name: Akshay Mehra
    affiliation: "1"
    orcid: 0000-0002-3966-1469
  - name: Arianne Ducellier
    affiliation: "1"
    orcid: 0000-0003-3668-9455
  - name: Ziheng Sun
    affiliation: "2"
    orcid: 0000-0001-9810-0023
  - name: Stefan Todoran
    affiliation: "1"
    orcid: 0009-0001-4711-5126

    
affiliations:
 - name: University of Washington, Seattle, USA
   index: 1
 - name: George Mason University, George Mason, USA
   index: 2
date: January 10, 2025
bibliography: paper.bib


---


# Summary

The "Machine Learning in the Geosciences" course, offered as ESS 469/569 at the University of Washington since 2023 and up the time of publication, introduces both undergraduate and graduate students to the application of machine learning (ML) techniques within geoscientific contexts. 

# Statement of need 

Machine learning (ML) has rapidly emerged as a transformative tool in the analysis of big data and scientific discovery across disciplines, especially since 2010. Geosciences, with its inherently large, complex, and multidimensional datasets, is particularly poised to benefit from ML's capabilities [@bergen2019, ]. Yet, despite the explosion of ML applications in geoscientific research, there is no established curriculum in higher education that focuses on equipping students with practical ML skills tailored to the unique needs of geosciences.  

While data science courses have recently gained traction, they typically provide general-purpose training that lacks the domain-specific emphasis critical for addressing the challenges of geoscientific datasets, such as handling spatiotemporal structures, geospatial data formats, and the integration of physical constraints into ML models. A course dedicated specifically to ML in geosciences bridges this gap, ensuring students and researchers gain the expertise to tackle pressing environmental and Earth system challenges through AI-driven approaches.  

The **JupyterBook** created for this course is particularly timely. Geoscience programs across institutions are increasingly recognizing the critical importance of AI and ML research. However, these programs often lack the resources or infrastructure to independently develop practical, cutting-edge ML curricula. This JupyterBook provides an accessible, open-source, and modular framework that can be immediately integrated into academic programs, accelerating the adoption of AI technologies within geoscientific education and research.  

By offering hands-on, practical experience with ML techniques using geoscientific examples, this course ensures that students not only understand ML concepts but can also directly apply them to real-world problems. This foundational training is vital for preparing the next generation of geoscientists to leverage AI for critical discoveries, from climate change mitigation to natural hazard forecasting and resource exploration.  

In summary, this course addresses a growing need in higher education by filling a critical gap in geoscientific training. It equips students with ML expertise, fosters interdisciplinary innovation, and ensures geoscientific programs remain at the forefront of scientific discovery in the era of AI.


## The Story

The MLGEO course arose from merging a development course "Data Sciences in the Earth and Planetary Sciences" (2021) and the NSF funded Geosmart project [@Cristea2024]. The course became a senior undergraduate and graduate level course designed for students at the University of Washington primarily enrolled in Earth and Space Sciences, Atmospheric and Climate Sciences, Oceanography, Forestry, Fisheries, Civil Environmental Engineering, programs that focus on domain-specific science typically with vast amount of data and needed a practicum in machine learning. The course was reviewed by colleagues in applied mathematics and computer sciences program to differentiate between "applied machine learning" and "fundamental machine learning" in 2023. The course is now offered yearly and enrolls 35-40 students.



**Course Structure:**
The course has three pillars:

1. **AI-ready GeoData:** Focuses on geoscientific data modalities, characteristics, feature extraction, dimensionality reduction, and preparing datasets for AI applications.

2. **Classic Machine Learning:** Covers model training, evaluation, and robust training practices for algorithms such as K-means, random forests, and k-nearest neighbors.

3. **Deep Learning:** Explores foundational concepts including fully connected layers, convolutional neural networks, sequence-to-sequence learning with recurrent neural networks, and modern topics like physics-informed neural networks and network architecture search.

**Technical Skills Development:**

The course emphasizes building competencies in:

- **Shell scripting**

- **Version control with Git and GitHub**

- **Generative AI**, integrating GenAI for software development and literature synthesis.

- **Python programming**, utilizing packages such as NumPy, Pandas, scikit-learn, PyTorch

- **Data visualization** using Matplotlib, seaborn, Plotly

- **High-performance computing** strategies for cloud, HPC and 

**Prerequisites:**

Students should have completed courses in mathematics, applied mathematics, statistics. and possess some programming experience (e.g., CS160 or CS163). While prior knowledge of Python is recommended, the course provides refreshers on computing as needed.

For more detailed information, including access to tutorials and homework assignments, the course utilizes a dedicated GitHub repository:  

# Learning objectives

By the end of the course, students are expected to:

- Demonstrate proficiency in Python programming, Jupyter notebooks, Git version control, integration of GenAI in coding practices (e.g., copilot), conda environments, containers, and deploying software on new platforms.

- Construct a standard ML workflow that follows community best practices, including data preparation, model design, training, validation, and evaluation.

- Implement data manipulation strategies pertinent to geosciences, such as handling time series and geospatial data, data visualization, dimensionality reduction, and feature engineering.

- Understand and apply open science principles, ensuring reproducibility and adherence to digital scholarship standards.

- Gain familiarity with canonical examples of ML across various geoscience disciplines and identify strategies for using ML in geoscience when given a context for data richness, physical models, and problem set up.

- Evaluate the robustness of the machine learning pipeline utilized in scientific literature


The Book contains about 50 hours of instructional hours.


# Teaching materials

The class alternates between notebooks, slides, and student led presentations.

## Detailed syllabus


## Slides

The majority of the class goes through the notebooks in the [Jbook](https://geo-smart.github.io/mlgeo-book/about_this_book/about_this_book.html), but several slidedecks are added for the convenience of the instructor. The GitHub repository contains raw materials for future instructors to adapt.

* [Introduction class](../book/slides/MLGeo_Introduction_generic.pptx): overview of ML in the geosciences, scientific concepts, course logistics.
* 


## Small Geoscientific DataSets

We collected a small data sets of geoscientific data that are used in the book, in class, or suggested by students participating in the class. The GitHub repository MLGEO-data (https://github.com/UW-MLGEO/MLGeo-dataset) sets contains a collection of notebooks (``./scripts/``)that demosntrates how to manipulate or download open-source data sets, and small data sets (typically CSV under ``./data/``) that are small enough to be used in the classroom.
```
git clone https://github.com/UW-MLGEO/MLGeo-dataset
```

## Docker Base Container
We created a minimal docker image to run the notebooks in class, automatically build with GitHub action from this repository: https://github.com/UW-MLGEO/MLGeo-image.

The image can be pulled with Docker:
```
docker pull uwessds/mlgeo-image:latest
```

## JupyterBook (Jupyter notebooks)

The MLGEO book is presented as a collection of Jupyter notebooks, organized into a Jupyter Book. This format allows for an interactive learning experience, where students can run code cells, visualize data, and experiment with different machine learning models directly within the notebooks.

The Jupyter Book is hosted online and can be accessed through the following link: [MLGEO Jupyter Book](https://geo-smart.github.io/mlgeo-book/). Each chapter is divided into multiple sections, with detailed explanations, code examples, and exercises to reinforce the concepts covered.

## Homework

The "Classic Machine Learning" [homework](https://geo-smart.github.io/mlgeo-book/Chapter3-MachineLearning/Homework_CML.html) assignment is designed to reinforce students' understanding of key machine learning concepts introduced in Chapter 3 of the course. The primary objective is to provide hands-on experience in data preparation, unsupervised clustering, and the application of various supervised learning algorithms.

In the initial phase, students engage in data preparation, which includes reading, cleaning, exploring, and reducing the dimensionality of a dataset. This process ensures that students can effectively handle real-world geoscientific data, making it suitable for machine learning applications. Subsequently, students apply unsupervised clustering techniques, specifically K-Means, to identify inherent patterns within the data. This step emphasizes the importance of selecting optimal cluster numbers and evaluating clustering performance.

The assignment culminates with the implementation of various supervised learning models, such as K-Nearest Neighbors, Naive Bayes, Random Forest, Support Vector Machine, and Multi-Layer Perceptron. Students are tasked with feature scaling, splitting data into training and testing sets, designing models, and evaluating their performance using metrics like confusion matrices and cross-validation. This comprehensive approach ensures that students gain practical skills in model selection, training, and evaluation, directly applying the theoretical concepts covered in Chapter 3.


## Final Project

The final project a group-based, complete machine learning project with 4 pillars:
1. Design a scientifically sound machine learning project, and justify the need and benefit of using ML. Establish the best non-ML approach to solving the problem and the baselines to beat.
2. AI-ready data set: 
    - explore data and its dimensionality
    - ML data pipelines
    - curated and explored data sets
3. Classic Machine Learning: develop baseline models using simple CML models, performa auto-ML to find optimal CML model
4. Deep Learning: explore deep learning models and their architectures, and if it improves upon the CML performance, develop comprehensive comparison of CML, DL, and argue for benefit of one or all approach with computational considerations.
The final proejct is described in the book: https://geo-smart.github.io/mlgeo-book/Chapter1-GettingStarted/1.20_MLGEO_Final_Project.html.



## Technology Integration  

The course emphasizes building a robust technological foundation for students to succeed in applying machine learning to geosciences. In the first week, students are introduced to generative AI tools for coding, such as GitHub Copilot, to accelerate their ability to draft and refine code efficiently. A significant focus is placed on ensuring students have access to appropriate software platforms, including setting up VSCode, creating GitHub accounts, and installing either a pre-configured Docker image or a Conda environment tailored for the course. Students are guided step-by-step to establish a well-organized workspace, integrating VSCode with Copilot for seamless AI-assisted coding. These sessions also cover best practices for managing environments, troubleshooting installations, and maintaining reproducibility in their workflows. By mastering these tools early in the course, students are empowered to tackle coding challenges with confidence and efficiency, leveraging cutting-edge AI technologies to enhance their productivity and technical skills.

Some of the notebooks presented in the JupyterBook were also supported by GenAI in order to craft new geoscientifically inspired toy synthetic data sets for in-class exercises. We found that GenAI was helpful in smoothing over notebooks, crafting new interesting exercises, expanding the instructor's perspective on interndisciplinary research.


# Content Delivery

The course is structured to provide a balanced and engaging learning experience, with each week designed to focus on three key components: 1/3 conceptual understanding, 1/3 application through toy problems, and 1/3 hands-on student-led exercises. This structure ensures that students not only grasp the theoretical aspects of machine learning but also apply them in practical scenarios and take an active role in the learning process.  

Weekly student participation includes presenting summaries of scientific papers or webinars, with an emphasis on selecting five presentations per week to encourage peer learning and collaborative discussions. The overall course is organized into three major pillars: 1/3 on data curation, 1/3 on classic machine learning methods, and 1/3 on deep learning techniques. Assignments, mostly tackled in groups, align with these pillars and culminate in a final group project that integrates all learned components. Additionally, an extra homework assignment helps assess individual learning outcomes, ensuring that students achieve a comprehensive understanding of the material.  

Students are provided ample opportunities to practice during class, fostering collaborative problem-solving and real-time feedback. The course is well-suited for remote delivery with its reliance on digital tools like Jupyter notebooks, GitHub, and cloud computing platforms. However, successful remote implementation requires additional teaching assistants (TAs) and breakout room support to address diverse student needs effectively. This interactive and flexible approach ensures that students are well-prepared to tackle complex geoscientific problems using modern machine learning techniques.

# Teaching experience

The course is designed for an instructor and a teaching assistant. Instructors tend to be among a single subdiscipline of geosciences and the student attending the course have been largely coming from geophysics, atmospheric sciences, oceanography, forestry, etc. Students have typically be 50% from undergraduate (senior level) and 50% from graduate students.

Classes were taught with 90 minutes class time three times a week, student spend several hours per week on their final project, paper reviews, and homework.

Instructors have provisioned jupyterhub for the class, with the mlgeo-image as base image. In the 2024 course offering, we made the students install their environment locally with Visual Studio Code, a student license for GitHub education that included a free license to GitHub CoPilot, and integrated this to the instructional time. Students downloaded the Jbook's notebook on their local Mac, Linux, and PC laptops, and ran the notebooks locally. It tooks a full week to have all 35 students fully ready to run the notebooks. 

The integration of GenAI in the 2024 course offering was transformative: the instructor spent less time debugging in class, more time discussing ML concepts, the students spent less time stuck on software engineering and formating, more time discussing their data. This acceleration enable students's final project to complete the 3 milestones of the class: AI-ready data, CML, and DL.


## Conclusion and Outlook

Overall, the enhanced teaching experience fostered a more interactive and productive classroom environment, ultimately leading to a more comprehensive understanding of machine learning principles and their practical applications.

The Jbook is designed to be a dynamic document that the community is invited to contribute to. There is a lot to improve on bringing geoscientific data sets relevant exercises, improving on the concepts, and the ever evolving literature.

# Acknowledgments

We acknowledge the UW eScience institute support provided through office hours and support to the GeoSMART project. Part of this project was supported by the College of the Environment, and NSF GEOSMART award number OAC-2117834.


# References