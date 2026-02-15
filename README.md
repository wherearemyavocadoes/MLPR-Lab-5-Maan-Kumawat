# MLPR Lab 5 – Distance-Based Face Clustering

**Name:** Maan Kumawat  
**Roll No:** U20240010  
**Course:** Machine Learning and Pattern Recognition  
**Semester:** Spring 2026  

---

## Overview

This lab is about working with distance-based methods and images. We have to detect faces in an image, then cluster the detected faces using the K-Means algorithm, and classify a template image using the learned clusters.

This lab is about understanding distance-based methods and their application to a classification problem.

---

## Aim

- Detect faces using the Haar Cascade classifier.
- Extract the Hue and Saturation features of the detected faces.
- Perform K-Means clustering.
- Classify a template image using one of the learned clusters.
- Display the clusters learned by the algorithm.

---

## Methodology

### Face Detection

The image is converted to grayscale and processed using the Haar Cascade Classifier from the OpenCV library. The faces are highlighted using bounding boxes.

<img width="794" height="559" alt="image" src="https://github.com/user-attachments/assets/7bca81a9-8bbe-41be-8417-72bc1ff089cb" />

### Feature Extraction

The features from the detected faces are extracted by converting the faces to HSV color space. The average value of the Hue and Saturation values are calculated.

### K-Means Clustering

The K-Means algorithm is applied to the extracted features with k=2 clusters. The faces are grouped based on the similarity between the Hue and Saturation values.

<img width="1005" height="547" alt="image" src="https://github.com/user-attachments/assets/114fa45f-c5b0-48f0-bad9-98dc166e4a01" />

The centroids are calculated and plotted to determine the separation between the clusters.

<img width="1005" height="547" alt="image" src="https://github.com/user-attachments/assets/536247eb-c057-455b-b2f6-ce896388822e" />

### Template Classification

The template image is converted to HSV color space and features are extracted. The cluster label is predicted using the trained K-Means model.

<img width="389" height="411" alt="image" src="https://github.com/user-attachments/assets/22305344-2fcf-4646-ae92-70bf909b7520" />

---

## Distance Metrics Discussed

- Euclidean Distance  
- Manhattan Distance  
- Minkowski Distance  
- Chebyshev Distance  
- Mahalanobis Distance  
- Cosine Distance  
- Hamming Distance  

---

## Bias-Variance in KNN

- Small K means Low bias, High variance (overfitting)  
- Large K means High bias, Low variance (underfitting)  

Cross-validation can be employed to determine an optimal K to trade off between bias and variance.

---

<img width="1005" height="547" alt="image" src="https://github.com/user-attachments/assets/f77326e8-63d0-4f8c-aa5f-95b3fe3371dd" />
<img width="1005" height="547" alt="image" src="https://github.com/user-attachments/assets/32fa6334-10fc-4df5-ad96-7d2305221c15" />



## Conclusion

In this lab, it is shown that distance-based methods can be used to cluster and classify images. The concepts covered include feature extraction, clustering, and evaluation.

---

## Tools Used

- Python  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn

