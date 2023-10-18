# This chapter focuces on model workflow and ML reproducibility


Reproducibility using [Make](http://swcarpentry.github.io/make-novice/)

In the Geosciences, every process is not isolated and one change will be transmitted to the entire system. It also applies in the geospatial data analysis, where each process is just one component of the entire giant data processing workflow. The data products you received actually have went through many upstream processes already. Changing any process will impact the final data and alter the information you are reading. In the AI/ML use case, it is a general question that if one team published a paper, how another team can reproduce the same results using the described workflow. ML algorithms are sometimes embedded with random number generator (e.g., deep neural network), every training run will produce a slightly different version even though everything is the same as previous run. In that case, it would be significantly difficult for other people to produce the exact same results because of the introduced randomness. 