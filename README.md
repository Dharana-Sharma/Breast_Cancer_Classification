
# Breast Cancer Classification
[View our project](https://btcs-g01.streamlit.app/)

## Introduction

Breast cancer is a prevalent form of cancer among women. It is also the second leading cause of cancer-related deaths in women. Breast cancer occurs when cells in the breast tissue grow abnormally and form a tumor, which can be either benign, pre-cancerous, or cancerous. Various medical tests, such as MRI, mammogram, ultrasound, and biopsy, are commonly used to diagnose breast cancer. In machine learning, breast cancer diagnosis is considered a classification problem, where the prediction is categorized into either malignant or benign based on the discrete labels in the data.

## 1.1 Aim and Objective

The principal objective of this inquiry is to employ and contrast various machine learning models to discern and classify breast cancer into either benign or malignant manifestations.

## 1.2 Methodology

The dataset is subjected to a rigorous examination to identify any null values and purged accordingly. In the event that the classification data is encoded as string values, they are transformed into integer format. Once the data is sanitized, an in-depth analysis is performed to extract its characteristic features. Subsequently, the pristine data is bifurcated into two distinct sets, namely train and test datasets. Upon successful partitioning, the classifier model is executed, and the accuracy of the test dataset is predicted.

## Support Vector Machine
The support vector classifier is a classification model utilized for support vector machines. Support vector machines (SVMs) are supervised machine learning algorithms that can be applied to both classification and regression tasks. SVMs are defined as discriminative classifiers that determine the optimal separating hyperplane. In other words, given a labeled set of training data, the algorithm will produce an ideal hyperplane to categorize new examples. In two-dimensional space, this hyperplane is a line that separates a plane into two parts, with each class lying on either side. We can implement a SVM model by following the below steps-

```python
Step 1: Load the data
Step 2: Split data into train and test
Step 3: First, it finds decision boundaries that correctly classify the training dataset.
Step 4: Pick the decision boundary which has maximum distance from the nearest points (supported vectors) of these two classes as the best one.
Step 5: Predict values using the SVM algorithm model
Step 6: Calculate the accuracy and precision. 
```

## Logistic Regression

To estimate the likelihood of a particular class or occurrence, such as pass/fail, win/lose, alive/dead, or healthy/sick, logistic regression is utilized as a statistical model. It may also be expanded to encompass multiple classes of events, such as ascertaining whether an image contains a cat, dog, lion, and so on. The methodology of the Logistic Regression algorithm may be comprehended by following the steps enumerated below -

```python
Step 1: Load the data
Step 2: Logistic Regression measures the relationship between the dependent variable and the independent variables, by estimating probabilities using its underlying logistic function.
Step 3: These probabilities must then be transformed into binary values in order to actually make a prediction using the sigmoid function. 
Step 4: The Sigmoid-Function takes any real-valued number and map it into a value between the range of 0 and 1.
Step 5: This values between 0 and 1 will then be transformed into either 0 or 1 using a threshold classifier.
```
## Decision Tree

A decision tree classifier is a machine learning model that uses a tree-like structure to make decisions based on input features. The tree is built by recursively splitting the data based on the most informative feature, with the goal of minimizing impurity or maximizing information gain at each split. Once the tree is constructed, it can be used to classify new data by traversing the tree based on the input features and reaching a leaf node that corresponds to a particular class label.

```python
Step 1: Load the data
Step 2: Split data into train and test
Step 3: Train and validate the model. Evaluate the performance on the testing data by calculating metrics such as accuracy, precision, recall. 
Step 4: Visualize the decision tree to gain insights into the model. This can help identify overfitting and interpret the results.
```

## Random Forest

"Random forests" or "random decision forests" refer to a technique of ensemble learning utilized for classification, regression, and other related tasks. It operates by producing numerous decision trees during the training phase and generating the mode of the classes or the average prediction of the individual trees as the output. The working principle of the Random Forest algorithm can be comprehended by following the sequential steps enumerated below -

```python
Step 1: First, start with the selection of random samples from a given dataset.
Step 2: Next, this algorithm will construct a decision tree for every sample. Then it will get the prediction result from every decision tree.
Step 3: In this step, voting will be performed for every predicted result.
Step 4: At last, select the most voted prediction result as the final prediction result.
```
## Naive Bayes

A Naive Bayes classifier is based on the Bayes theorem, which describes the probability of a hypothesis based on prior knowledge and evidence. The Naive Bayes classifier assumes that the features are independent of each other, which makes it a simple and efficient model. It works by calculating the probability of each class label given the input features and choosing the label with the highest probability as the prediction. The model is trained on labeled data by estimating the probabilities of the features given each class label. Steps to implement NB Classifier are as follows:

```python
Step 1: Load the data
Step 2: Split data into train and test
Step 2: Use the training data to estimate the probability of each feature given each class label using Bayes Theorem.
Step 3: Evaluate the performance of model on the testing data by calculating metrics such as accuracy, precision, recall.
Step 4: Visualize the model to gain insights into the model and its decision-making process.
Step 5: Use the Naive Bayes model to predict the class labels of new data based on its features. 
```

## 1.3 Software Requirements
Jupyter Notebook: Jupyter Notebook is an essential tool for machine learning projects as it enables the integration of code, data, and visualizations in one platform. It allows for the creation and sharing of documents containing live code, equations, and narrative text, making it an excellent choice for prototyping, experimentation, and collaboration. Jupyter Notebook supports many programming languages, including Python, R, and Julia, making it a versatile tool for machine learning projects.

## 1.4 Tool Description

NumPy - NumPy is a Python library that enables support for large, multi-dimensional arrays and matrices, and includes a vast collection of high-level mathematical functions designed to operate on these arrays.

pandas - pandas is a Python software library primarily designed for data analysis and manipulation. It provides powerful data structures and operations for working with numerical tables and time series data.

Matplotlib - Matplotlib is a plotting library for Python that can be used in conjunction with NumPy to create high-quality, customizable visualizations. It offers an object-oriented API for integrating plots into applications that use general-purpose GUI toolkits like Qt, wxPython, GTK+, or Tkinter.

SVC - Support Vector Classifier is a classifier model that is part of Support Vector Machine (SVM). The SVM algorithm is a supremely versatile supervised machine learning tool that can be used to solve both classification and regression challenges. 

DecisionTreeClassifier - A decision tree classifier is a machine learning algorithm that uses a tree-like model to make predictions or decisions based on the input features. The tree structure represents a sequence of decisions and their possible outcomes, leading to a final prediction for the input data.

RandomForestClassifier - This algorithm is a ensemble learning method that excels in classification, regression, and other tasks. During its training phase, it concocts a multitude of decision trees, which then output the class that represents the mode of the classes or mean prediction of the individual trees. 

LogisticRegression - The LogisticRegression algorithm utilizes the logistic model to estimate the probability of a particular class or event's occurrence. It can be applied to a wide range of scenarios, from pass/fail, win/lose, alive/dead, or healthy/sick classifications, to the detection of multiple classes of events such as identifying the presence of various animals like cats, dogs, or lions in an image.

NaiveBayes - The Naive Bayes classifier is a probabilistic machine learning algorithm that utilizes Bayes' theorem and the assumption of independence between features to classify data. It calculates the probability of each class for a given input and selects the one with the highest probability as the predicted class.

## Implementation

## 2.1 Design & Implementation

To design and implement a Breast Cancer Classification system using machine learning techniques and the Wisconsin Breast Cancer dataset, we first preprocessed the data by performing feature selection and normalization. We use various classification algorithms like Decision Trees, Logistic Regression, and Support Vector Machines to build and train the model. The model's performance can be assessed using metrics such as accuracy, precision, and recall. Accuracy is the percentage of correctly classified cases, precision represents the percentage of true positives among predicted positives, and recall measures the percentage of actual positives identified by the model. Finally, we fine-tune the model's parameters and deploy it for real-world use to achieve efficient and accurate diagnosis of breast cancer.

## 2.1.1 Implementation Mechanism
The preparation of data is a critical aspect of any data analysis task, as it enables one to reveal the underlying structure of the problem for the machine learning algorithms that are to be utilized. An ideal approach is to employ pandas to import the data, and subsequently remove any unnecessary columns such as ID that hold no relevance to the model. It is also important to scan the dataset for any null values and subsequently eliminate them from the rows to ensure that the data is accurate and consistent.

## 2.1.2 Major Considerations for Implementation
One of the challenges encountered was that the "class" attribute values were expressed as strings, hence the need to convert them to integers in order to serve as our target class. We were able to achieve this by utilizing the LabelEncoder function provided by the sklearn library.

## Results & Discussion
After much research and analysing previous research papers as well , we found that for our proposed study:
-Logistic Regression Model gave the best result – an accuracy of 98.24% on the whole dataset i.e. considering all the 30 attributes.
-Random Forest Model gave the best result – an accuracy of 97.36% on consideration of the worst valued attributes i.e. considering only 10 attributes.

Further , we have also deployed our [Breast Cancer Classification](https://btcs-g01.streamlit.app/) Project to check the malignancy of the tumour and predict whether the tumour is cancerous or not.



## Conclusion
```python
1. The Objective of the project was to predict whether the tumour is malignant or benign.
2. Different Machine Learning algorithms were used to make prediction using the Wisconsin Breast Cancer dataset.
3. Logistic Regression provided highest accuracy of 98.24%.
4. While Decision Tree Classifier gave the worst accuracy of 94.74% for prediction.
Upon standardizing data, we get good accuracy results using Support Vector Machine, Logistic Regression, Random Forest Decision Tree and Naive Bayes, correctly classifying tumour into malignant or benign almost 95-98% of times.

```













