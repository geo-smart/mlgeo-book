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

1. **AI-ready GeoData:** Focuses on geoscientific data modalities, characteristics, feature extraction, dimensionality reduction, and preparing datasets for AI applications.

2. **Classic Machine Learning:** Covers model training, evaluation, and robust training practices for algorithms such as K-means, random forests, and k-nearest neighbors.

3. **Deep Learning:** Explores foundational concepts including fully connected layers, convolutional neural networks, sequence-to-sequence learning with recurrent neural networks, and modern topics like physics-informed neural networks and network architecture search.

**Learning Objectives:**

By the end of the course, students are expected to:

- Demonstrate proficiency in Python programming, Jupyter notebooks, Git version control, and deploying scripts on various computing platforms.

- Develop and apply standard ML workflows, including data preparation, model design, training, validation, and evaluation.

- Implement data manipulation strategies pertinent to geosciences, such as handling time series and geospatial data, data visualization, dimensionality reduction, and feature engineering.

- Understand and apply open science principles, ensuring reproducibility and adherence to digital scholarship standards.

- Gain familiarity with canonical examples across various geoscience disciplines.

- Comprehend advanced techniques like Fourier and wavelet transforms, principal component analysis, and their applications in data interpretation.

**Technical Skills Development:**

The course emphasizes building competencies in:

- **Shell scripting**

- **Version control with Git and GitHub**

- **Python programming**, utilizing packages such as NumPy, (Geo)Pandas, scikit-learn, Keras, and PyTorch

- **Data visualization** using Matplotlib and Plotly

- **High-performance computing** strategies

**Prerequisites:**

Students should have completed courses in mathematics (MATH 207 and MATH 208, or equivalents) and possess some programming experience (e.g., CS160 or CS163). While prior knowledge of Python is recommended, the course provides refreshers on computing as needed.

For more detailed information, including access to tutorials and homework assignments, the course utilizes a dedicated GitHub repository:  


For learning modules, describe the learning objectives, content, instructional design, and experience of use in teaching and learning situations.

# Learning objectives



The Book contains about 50 hours of instructional hours.
# Teaching materials

## Detailed syllabus


## Slides

## Docker Base Container

## JupyterBook (Jupyter notebooks)

## Homework

The "Classic Machine Learning" [homework](https://geo-smart.github.io/mlgeo-book/Chapter3-MachineLearning/Homework_CML.html) assignment is designed to reinforce students' understanding of key machine learning concepts introduced in Chapter 3 of the course. The primary objective is to provide hands-on experience in data preparation, unsupervised clustering, and the application of various supervised learning algorithms.

In the initial phase, students engage in data preparation, which includes reading, cleaning, exploring, and reducing the dimensionality of a dataset. This process ensures that students can effectively handle real-world geoscientific data, making it suitable for machine learning applications. Subsequently, students apply unsupervised clustering techniques, specifically K-Means, to identify inherent patterns within the data. This step emphasizes the importance of selecting optimal cluster numbers and evaluating clustering performance.

The assignment culminates with the implementation of various supervised learning models, such as K-Nearest Neighbors, Naive Bayes, Random Forest, Support Vector Machine, and Multi-Layer Perceptron. Students are tasked with feature scaling, splitting data into training and testing sets, designing models, and evaluating their performance using metrics like confusion matrices and cross-validation. This comprehensive approach ensures that students gain practical skills in model selection, training, and evaluation, directly applying the theoretical concepts covered in Chapter 3.

## Technology Integration  

The course emphasizes building a robust technological foundation for students to succeed in applying machine learning to geosciences. In the first week, students are introduced to generative AI tools for coding, such as GitHub Copilot, to accelerate their ability to draft and refine code efficiently. A significant focus is placed on ensuring students have access to appropriate software platforms, including setting up VSCode, creating GitHub accounts, and installing either a pre-configured Docker image or a Conda environment tailored for the course. Students are guided step-by-step to establish a well-organized workspace, integrating VSCode with Copilot for seamless AI-assisted coding. These sessions also cover best practices for managing environments, troubleshooting installations, and maintaining reproducibility in their workflows. By mastering these tools early in the course, students are empowered to tackle coding challenges with confidence and efficiency, leveraging cutting-edge AI technologies to enhance their productivity and technical skills.

Some of the notebooks presented in the JupyterBook were also supported by GenAI in order to craft new geoscientifically inspired toy synthetic data sets for in-class exercises. We found that GenAI was helpful in smoothing over notebooks, crafting new interesting exercises, expanding the instructor's perspective on interndisciplinary research.


# Content Delivery

The course is structured to provide a balanced and engaging learning experience, with each week designed to focus on three key components: 1/3 conceptual understanding, 1/3 application through toy problems, and 1/3 hands-on student-led exercises. This structure ensures that students not only grasp the theoretical aspects of machine learning but also apply them in practical scenarios and take an active role in the learning process.  

Weekly student participation includes presenting summaries of scientific papers or webinars, with an emphasis on selecting five presentations per week to encourage peer learning and collaborative discussions. The overall course is organized into three major pillars: 1/3 on data curation, 1/3 on classic machine learning methods, and 1/3 on deep learning techniques. Assignments, mostly tackled in groups, align with these pillars and culminate in a final group project that integrates all learned components. Additionally, an extra homework assignment helps assess individual learning outcomes, ensuring that students achieve a comprehensive understanding of the material.  

Students are provided ample opportunities to practice during class, fostering collaborative problem-solving and real-time feedback. The course is well-suited for remote delivery with its reliance on digital tools like Jupyter notebooks, GitHub, and cloud computing platforms. However, successful remote implementation requires additional teaching assistants (TAs) and breakout room support to address diverse student needs effectively. This interactive and flexible approach ensures that students are well-prepared to tackle complex geoscientific problems using modern machine learning techniques.

# Teaching experience

The course is designed for an instructor and a teaching assistant. Instructors tend to be among a single subdiscipline of geosciences and the student attending the course have been largely coming from geophysics, atmospheric sciences.

# Acknowledgments

We acknowledge the UW eScience institute support provided through office hours and support to the GeoSMART project. Part of this project was supported by the College of the Environment, and NSF GEOSMART award number OAC-2117834.


# References