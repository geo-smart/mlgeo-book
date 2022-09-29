# Computing Environments

We will discuss on how to work with various environments

## Local environment. 

To work locally using python and Jupyter notebooks.

⚠️ this content only conerns Linux and MacOS, please reach out to contribute to windows environments.

We recommend using [Visual Studio Code](https://code.visualstudio.com/) for code and notebook editing, running codes and notebooks, editing github repository, interacting with GitHub. It is how we wrote this book!

Either through your OS or using VSCode, open a terminal. The terminal is used to pass commands to the computer. The programming language to send codes to the computer is Shell. BASH (Bourne Again SHell) is one of the implementations of Shell and is used to efficiently perform tasks. Great tutorials on how to get started with Shell are in the [Software Carpentries Lessons](https://swcarpentry.github.io/shell-novice/).


Get to know your hardware. Find out what CPU, GPU, what memory is on your local environment. One that allows to monitor the resources in real time is ``htop``. Install it on your local environment (e.g., use ``brew install htop`` on MacOSX). 

Machine Learning research requires parallelization. Typically, CPUs (Central Processing Units) are plenty sufficient for most ML applications. They are the basics and default hardware of most computers and are particularly practical when handling large data sets for the In/Out operations. 

 However, Deep Learning applications are mostly enabled by the much expansive parallelization of GPUs (Graphic Processing Units). GPUs were developed mostly for gaming, and it is why they have become affordable. The pros of GPU

```watch nvidia-smi```


The local environment is 
- What is a Cloud system (definition of an instance)
- What is an HPC system (definition of a node)
- What is a jupyterHub:
    - Free hubs: Google Colab, AWS Sage Studio, Microsoft Planetary Computer
    - UWIT hosted Hub.
- How to connect to AWS instances
- How to connect to Azure instance