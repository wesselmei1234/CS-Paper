# CS-Paper
This project is an assignment for the Econometrics master program from Erasmus University Rotterdam. It is part of the computer science for business analytics course. 

The project aims to perform scalable duplicate detection on a tv dataset. Duplicate detection methods might take a long time to converge. In this project Locality-Sensitive Hashing is applied as preprocessing in order to reduce the number of computations needed. 

To use the code, you should load all files and the data. Everything is tied to the solution.py file. You only need to set the parameters and then run the file. 

The tv.py file creates two classes which are used to make things easier later on. The tv class represents the tv products. Some get methods are defined and an equals method which later is used to check if two tvs have a similar modelID. The cleaned definition is used to clean the title in a way that model words can be extracted more easily. The file contains two other functions. The first creates a list of all tvs. The second cleanes some values of the key-value pairs which represent features of the tv. 

func.py declares the functions which are needed to run the project. The main function decleration is run iteration. This runs a complete iteration for one train set and test set. First, each tv is given a brand (if one can be found). The model words are then extracted from the title and the features. These serve as input for the binary vector representations. From these vectors a signiture matrix is created which is minhashed. This results in candidate pairs. These candidate pairs are tuples of two tvs which should be checked in the clustering step. A grid is specified over which the clustering algorithm should run and tune the hyperparameters. Finally, the process is repeated for the test set. This results in the performance measures. 

Clustering.py is the file which performs the clustering. First, some blocking heuristics are applied. Following, possible model IDs are extracted. If these modelIDS match, the distance is set to almost zero. After that, similarity measures are computed for the remaining pairs. Single linkage hierarchical clustering is then applied to find the duplicates. Finally the performance is evaluated in terms of F1, precision, and recall. 

Final conclusion is that the number of computations can be drastically reduced, while losing a little bit of performance. 
