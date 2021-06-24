# KNN-classifier-

In  this K-Nearest Neighbors code there is no learning required as the model stores the entire dataset and classifies data points based on the points that are similar to it. It makes predictions based on the training data only. 
the distance it uses to operate on is eaither Euclidean, Manhatten or MinKowski, feel free to uses other distance according to your need.
to classify the data accordingly into a class, we store the distance in an array and sort it according to the ascending order of their distances, then we select the first K elements in the sorted list. Finally , we perform the majority Voting and the class with the maximum number of occurrences will be assigned as the new class for the data point to be classified. 
