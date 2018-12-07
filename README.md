# Face_Social_Traits_and_Political_Election_Analysis_by_SVM

## Classification by Landmarks
For this part, we are only going to use the landmarks of 491 face images as the training data to train SVM/SVR models for 14 face traits. The landmarks contain 80 coordinate values. To use these values for SVM/SVR models, these coordinates are combined to be one vector with 160 x and y values. As a result, the data will be a 491x160 matrix. 30% of the data has been randomly chosen as the test data and the rest 70% is the training data.

For each face trait, there is a SVM/SVR model fitted. The functions used is the SVC and SVR from the SVM package in SK-Learn in Python. A K-fold cross validation has been done to choose the best parameters we want for each of the 14 models. Here I used a 5-fold cross validation. For SVM models, the parameters tuned are C (penalty) and gamma (for the rbf kernel). For SVR models, the parameters tuned are C, gamma and epsilon.

## Classification by Rich Features
We can consider the landmarks as the poor features. Now we want to imply more features of the images. Here we use the HOG (histogram of oriented gradient) features which can extract the main shapes and information of a face images such as the outlines of the face, eyes, noses and the mouth.

The vector of HOG features here has a length of 6272 and the training data combines the hog features and the landmarks. 30% of the data has been chosen to be the test set. However, we only have a very small amount of the data and a much larger amount of features. It is very possible that the overfitting problem would be huge.

## Direct Predicition by Rich Features
Here, we will use the rich features formed by landmarks and the HOG features to do the Rank SVM for the election outcomes. We could simply use the SVM model here with rbf kernel to do the Rank SVM by reforming training dataset. According the election results, we could form the win-lose pairs of those features and use Win features – Lose features and Lose features – Win features to form the +1 and -1 groups.

## Prediction by Face Social Traits
Here two layers of SVM models have been used. The first layer is the SVR model we trained for the rich features. We would directly use the model we've trained to do the prediction of face traits for governors/senators.

Then a second layer of LinearSVM (without the bias term) will be trained. We are using the predicted face traits from first layer and then do the same thing in the "Direct prediction by rich features" part to perform a Rank SVM but just using the LinearSVM here so that we could have the specific values of the coefficients of each face traits.