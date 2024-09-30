# Machine Learning in the Geosciences
The **GeoS**cience **MA**chine Learning **R**esources and **T**raining (GeoSMART) framework provides an educational pathway and a foundation in open source scientific ecosystems and progresses through general ML theory, toolkits, and deployment on Cloud computing.

This book is used in the course offered at the University of Washington: Machine Learning in the Geoscienes (AUTMN 2023 - ESS 469/569). The corresponding GitHub repository with notebooks for the tutorials and homeworks is [MLGeo](https://github.com/UW-ESS-DS/MLGeo-2023). Find the Docker image for the corresponding jupyter hub [MLGeo Image](https://github.com/UW-ESS-DS/MLGeo-image).

Instructors:

- Marine Denolle (mdenolle@uw.edu)
- Akshay Mehra (akmehra@uw.edu)

This project is supported by the GeoSMART team (Stefan Todoran, Nicoleta Cristea, Anthony Arendt, Scott Henderson, Ziheng Sun, Yiyu Ni, Akash Kharita).

# Overview
The course is intended to introduce Machine Learning in Geosciences, the basics of computing, and methodologies in applied machine learning. The course focuses on canonical and topical data sets in seismology, oceanography, cryosphere, planetary sciences, geology, and geodesy. The methods taught include unsupervised clustering, logistic regression, random forest, support vector machine, and deep learning.

# Student version
Some chapters will include code sections which students should attempt to write on their own first before viewing the solution. The student version of this book automatically clears sections marked by the instructor as student response, and can be accessed via the link below. 

[![Student Version](../img/student_version_badge.svg)](https://geo-smart.github.io/mlgeo-book/)

# Learning objectives 
By the end of the quarter, the students should be able to:

- Demonstrate computing skills in python, jupyter notebooks, Git version control, and deploy scripts on local computers, cloud-hosted hubs, or cloud instances.
- Develop and apply standard machine-learning workflows: 1) Data preparation, 2) Model design, 3) Model training, validation, and evaluation.
- Apply standard data manipulation strategies in the Geosciences: data types (time series and geospatial), data formats, data visualization, dimensionality reduction, and feature engineering.
- Describe and demonstrate the adoption of open science principles, science reproducibility, and digital scholarship.
- Describe the canonical examples in a breadth of disciplines in geoscience.
- Understand at least qualitatively how some of the advanced techniques (Fourier and wavelet transform, principal component analysis,…) manipulate and transform the data to interpret the output.


# Tiered learning
Because students do not need to be experts on all materials covered in this course, we divide each topic into three levels: **Beginner (Level 1)**, **Intermediate (Level 2)**, and **Advanced (Level 3)**. We tag these levels throughout the book. 

Note that we introduce and incoroporate data visualization concepts throughout the book.

# Prerequisites
**Prerequisites**: MATH 207 and MATH 208, or MATH 307 or 308, or AMATH 351 or 352, CS160 or CS163, or permission from the instructor.

**Recommended skills**: Knowledge in Matlab or python, AMATH301, 100- or 200-level courses in the Earth Sciences. We will provide refreshers on computing as part of the course.

# Syllabus
- Module 1 (weeks 1 and 2): Intro on ML in the Geo and basic tool building for Open Sciences 
- Module 2 (weeks 3 and 4): Creating Machine-Learning Ready Data Sets
- Module 3 (weeks 5 and 6): Feature extraction and clustering
- Module 5 (weeks 7 and 8): Machine Learning
- Module 6 (weeks 9 and 10): Deep Learning

# Technical skills building
Throughout the course, the students will build skills in shell, version control using git and GitHub, Python programming, high-performance computing strategies, and simple data visualization using Python. 
- _Shell_: introduced early in the course, and manipulated if needed
- _Version Control_: introduced early in the course and used at every lecture
- _Python Programming_: progressively introduced. Specifically, we detail the use of the following packages: numpy, (geo)pandas, sklearn, keras, pytorch.
- _Visualization in Python_: introduced early as Matplolib and Plotly, and used in every Python lecture.
- _High performance computing_: used in the second half of the course and during development of the final project.

# Readings and webinars
Each week, students will write a short report about either a paper or a webinar. Use the template on canvas and answer the questions when appropriate. Submissions of the report PDF are due Wednesdays at 11:59 pm PDT on canvas. The instructor will spend 15 minutes Monday morning summarizing the reading and webinar reports. Papers can be found and/or uploaded on a shared private course Google Drive [here](https://drive.google.com/drive/folders/1dyxfslCLzFFTYtX_vbjudlzaXvOxkepe?usp=sharing) (only accessible with a UW email address).

# Github with tutorials and homeworks --- Course specific
The course [GitHub](https://github.com/UW-ESS-DS/MLGeo-2023) contains the tutorial notebooks. Clone the tutorial

        git clone "https://github.com/UW-ESS-DS/MLGeo-2023"

To update the local repository from the remote version
        git fetch
        git merge
        git pull

To force-reset the repository from the main branch
        git reset --hard origin/main
        git pull

Make your own repository (MLGEO2023_UWNETID). Copy the environment.yml file and the tutorials into your own reposistory to run and modify them.