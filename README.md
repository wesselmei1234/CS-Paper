# CS-Paper

This project aims to perform scalable duplicate detection on a tv dataset. Duplicate detection methods might take a long time to converge. In this project Locality-Sensitive Hashing is applied in order to reduce the number of computations needed. 

To use the code, you should load all files and the data. Everything is tied to the solution.py file. You only need to set the parameters and then run the file. 

The tv.py file creates two classes which are used to make things easier later on. The tv class represents the tv products. Some get methods are defined and an equals method which later is used to check if two tvs have a similar modelID. The cleaned definition is used to clean the title in a way that model words can be extracted more easily. The file contains two other functions. The first creates a list of all tvs. The second cleanes some values of the key-value pairs which represent features of the tv. 

