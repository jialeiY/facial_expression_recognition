# facial_expression_recognition
## Overview
This project implemented a facial expression recognition system to 
classify 6 types of emotions using the Cohn-Kanade Database.

In the project, PCA and Gabor wavelet were applied for feature extraction. 
PCA was used on the entile face area and the Gabor wavelet were only used 
on eyes and month areas. A K-nearest neighbors (KNN) classifier was trained 
to classifify extracted features.

## File Description
1.'main.m': main entrance if the project
2.'PCA.m': function for PCA
3.'gaborKernel.m': build kernel for Gabor wavelet transformation
4.'GaborTrans.m': implement gabor wavelet on specific area using the built kernel
5.'KNN.m': function of KNN classifier
6.'preprocessing.m': preprocessing of the images, includes: downsampling and facial
parts recognition
7.'subsample.m': function of subsampling
