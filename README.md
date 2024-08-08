# Introduction/Background

Machine learning aids in predicting Alzheimer's disease in its early stages. Vector machines, decision trees, random forest, and CNNs are some methods used [3]. Studies have taken a computer vision approach, wherein CNNs are used to detect abnormalities in MRI and PET scans of the brain. Classifiers are also used to diagnose patients into Alzheimer's vs healthy vs mild cognitive impairment (MCI), where a balanced accuracy of 90.6% was achieved [1]. Other techniques involve using data from psych evaluation and common brain tests. Regression and classification analyses are used to understand the distribution of features, often reaching over 80% accuracy. [2]

The Alzheimer's Prediction Dataset includes information such as age, education, socioeconomic status, and exam scores. The 'Group' column serves as the target variable, to help us train the algorithm to classify a sample patient into an Alzheimer's vs non-Alzheimer's group.

## Dataset Link:
https://www.kaggle.com/datasets/brsdincer/alzheimer-features

# Problem Definition
We aim to develop a method to identify Alzheimer's Disease (AD) since the exact cause is undetermined. Research has shown that machine learning algorithms can be used to efficiently weigh features based on their likelihood in existing in AD patients. Our algorithm hopes to reach 80%> accuracy in diagnosis in order to be considered as a potential intervention before further development of the disease. 

# Methods
## Data Preprocessing Method Implemented

Data Preprocessing:

We first preprocessed the raw dataset by deleting rows that had 6 missing cells or more. This threshold was decided based on previous studies (Bogdanovic et. al, 2022) that used similar Alzheimer’s/Dementia datasets. By removing these rows, we can ensure that our analysis is based on more reliable and complete data.
![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/Screenshot%202024-03-30%20at%205.55.42%20PM.png?raw=true)

For the patients that still had empty values for certain features, we used an average interpolation technique. This first finds data points in the same labeled category, and calculates the mean value for that feature with the missing data. The empty cell is then replaced with the mean value.

![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/Screenshot%202024-03-30%20at%205.56.49%20PM.png?raw=true)

To run our supervised model, we also had to convert categorical variables into numerical values. More specifically, we changed males to ‘0’, and females to ‘1’, and the target labels as 0 to 4, based on the severity of the neurodegenerative disease (“CN” = 0, “EMCI” = 1, “LMCI” = 2, “SMCI” = 3, “AD” = 4).
Once we had a complete, comprehensible dataset, we needed to reduce our 18 selected features to smaller, low-dimensional data. 

We were first curious to see what features are highly correlated between each other. We created a covariance heatmap using the Seaborn package to visualize which features are more important, and which are redundant to the dataset.

![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/heatmap.png?raw=true)

We then ran a Principal Component Analysis (PCA) for dimensionality reduction.  PCA allows us to reduce the number of features in our dataset while preserving most of the variance in the data. This is good for speeding up computation, reducing overfitting, and simplifying interpretation. 
We first created a principal components chart that indicates the amount of components necessary to maintain 80% variance.

![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/pca_chart.jpeg?raw=true)

We then used the sklearn package to run a PCA using 7 components based on the chart above. 

![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/Screenshot%202024-03-30%20at%206.45.30%20PM.png?raw=true)

Our reduced dataset included the features “AGE”, “PTEDUCAT”, “APOE4”, “ADS11”, “WholeBrain”, “TAU”, and “Entorhinal”. Below is an example subset of what our data looked like, including our target “DX_bl” and “PTID” as meta.

![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/Screenshot%202024-03-30%20at%206.46.53%20PM.png?raw=true)


## ML Algorithm Implemented

We initially chose to try a linear regression to predict the level of dementia...

However, while you can use a LR for predicitng multi-class data, we would need to define several thresholds to determine at what sections in the regression line would correlate to each label.

Therefore, we chose to try Naive Bayes.

### Model #1 - Naive Bayes:

Why?
- Works well for multi-class classification (We have 5:“CN”, “EMCI", “LMCI”, “SMCI”, “AD”)
- Is simple to interpret and quick to compute
- By running a PCA first, we de-correlate the features and help uphold the independent features assumption
- Does not typically require a lot of training data

And here is how we did it!


![alt_text](https://github.com/felolivee/AlzheimersDetection/blob/main/Screenshot%202024-03-30%20at%207.06.18%20PM.png?raw=true)

While we were happy with our performance, we wanted to try a few more different models.

### Model #2 - Multinomial Logistic Regression:
- No assumptions: LRs do not assume a specific distribution for the input features unlike some other classification algorithms
- Regularization: We can use Ridge regression or Lasso regression to prevent overfitting and improve generalization performance
![alt_text](https://github.com/felolivee/AlzheimersDetection/blob/main/Screenshot%202024-04-19%20at%204.46.47%20PM.png)

Some specifics:
- Uses l2 Ridge regression for regularization
- 1000 max iterations

### Model #3 - Neural Network:
- Captures non-linearity: Alzheimer’s is a complicated disease (hence why we have no cure for it!). It is possible that classification depends on non-linear combinations of the features. 
- Large Dataset: As the OASIS data expands with more patient samples, a neural network can better capture and generalize to unseen data
![alt_text](https://github.com/felolivee/AlzheimersDetection/blob/main/Screenshot%202024-04-19%20at%204.42.53%20PM.png)

Some specifics:
- 2 Hidden layers, with 7 neurons in each (7,7)
    - this comes from the fact that we are using 7 principal components
- Logistic activation function to capture nonlinearity
- Stochatic gradient descent for optimization
    - alpha of 0.0001 for step-size
- 1000 max iterations

# Results and Discussion
## Visualizations
We used a Confusion Matrices and ROC Curves to test the validity of our project.

### Naive Bayes
#### Confusion Matrix
The confusion matrix shows how many predictions were correct and incorrect for each class in our Naive Bayes Model. The classes are represented by integers 0 - 4: 0 is CN, 1 is EMCI, 2 is LMCI, 3 is SMC, and 4 is AD. The main diagonal shows the number of predictions from the model that are correct for 0 - 4 classes: 88%, 79%, 81%, 88%, and 92%. As we can see, there were misclassified data points. This could be due to the dimension reduction that was performed on the data prior to being used in the model in that some information could have been lost. Another reason for these misclassifications is that more data is needed to train the model.
![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/confusion_matrix.png?raw=true)

#### Receiver Operating Characteristic (ROC) Curve
The ROC curve is another indicator of the performance of a classification model at all of the classification thresholds. As seen below, the model is separated based on the targets (the classification level of Alzheimer’s). The closer the curve the top left corner, the more accurate the prediction model is. This aligns with what the confusion matrix said because the model was mostly accurate.
![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/roc_curve_01.png?raw=true)
![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/roc_curve_23.png?raw=true)
![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/roc_curve_4.png?raw=true)

### Logistic Regression
#### Confusion Matrix
From the confusion matrix we can see higher accuracy among the CN(0) target with 77% and the LMCI(2) target with 77.8% accuracy. The average accuracy, including the accuracy of the other targets, show that logistic regression is not the best model to use with our data set.
![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/logr_c_matrix.png?raw=true)

#### Receiver Operating Characteristic (ROC) Curve
We can use an ROC curve as well for logistic regression to evaluate the performance of the model at all classification thresholds. Target class 4 has the best classification compared to the model predicting the other classes as seen by its ROC curve having the greatest area under the curve. The discrepency between classifying target 4 compared to the other targets can be explained by how multiclass logistic regression works. The one-vs-rest scheme treats one class as the target variable to calculate the probabilities of each feature, but this requires treating the remaining 4 targets as a group. The poor ROC curves, especially for target 1 and target 3, indicate that logistic regression is not well-suited to achieve our goal of predicting one’s classification of Alzeihmers because the calculated probabilities are not representative of the ground truth.
![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/ltar01.png?raw=true)
![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/ltar23.png?raw=true)
![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/ltar4.png?raw=true)

### Neural Network
#### Confusion Matrix
With Neural Networks we can look at the confusion matrix to show how accurate our predictions were for each class. For each class we can look at the prediction accuracy: CN(0) - 91.5%, EMCI(1) - 80.6%, LMCI(2) - 89.6%, SMC(3) - 80.0%, AD(4) - 85.9%. Comparing these numbers from the other models we tested the data with, it can be seen that the average accuracy is highest with Neural Network. This is further proved with our metrics. 
![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/nn_confusion_matrix.png?raw=true)

## Quantitative Metrics
### Naive Bayes:
**AUC:** 0.928\
**CA:** 0.732\
**F1 Score:** 0.738\
**Precision:** 0.751\
**Recall:** 0.732\
**MCC:** 0.652 

### Logistic Regression:
**AUC:** 0.777\
**CA:** 0.521\
**F1:** 0.452\
**Precision:** 0.470\
**Recall:** 0.521\
**MCC:** 0.348


### Neural Network:
**AUC:** 0.975\
**CA:** 0.863\
**F1:** 0.862\
**Precision:** 0.862\
**Recall:** 0.863\
**MCC:** 0.818

## Analysis Of Model
### Naive Bayes:
#### AUC - 0.928:
AUC measures the ability of our model to identify whether a patient falls within our target variable or not. A value closer to 0.5 indicates that the model is essentially randomly classifying. Whereas a value closer to 1 is indicative of informed predictions. Our AUC value of 0.928 suggests that our model has good classifying ability. 

#### CA - 0.732:
CA measures the proportion of correctly classified instances, which is important in the context of AD classification as you don’t want to misdiagnose an individual. We have 5 target variable categories: CN (Cognitive Normal), EMCI (Early Mild Cognitive Impairment), LMCI (Late Mild Cognitive Impairment), SMC (Significant Memory Concern) and AD (Alzheimer’s Disease). Our CA is .732, which means that our model correctly identifies the target variable 73.2% of the time, based on our predictions. 

#### F1 Score - 0.738, Precision - 0.751, Recall - 0.732:
The F1 Score is the mean of precision and recall. This accounts for false positives and false negatives which are critical in the context of AD classification. False positives when a model incorrectly predicts that a patient has AD, which leads to stress and unnecessary medical intervention. A false negative is detrimental as this can lead to delayed diagnosis and treatment. And immediate care is crucial for AD patients. Our value of 0.738 indicates a good balance between precision and recall in predicting AD in patients. To improve the precision, recall, and hence F1 score, we would improve our feature selection. This could be done by focusing more on variables that capture the the complex interdependencies in clinical data more, or with a more sophisticated approach to encoding categorical variables. 

### Logistic Regression:
#### AUC - 0.777:
We got an AUC of 0.777 for our logistic regression model. AUC provides a measure of how well the logistic regression model can distinguish between different stages of Alzheimer's disease and healthy controls. An AUC of 0.777 indicates a decent ability to discriminate between the classes, although there definitely room for improvement. This is especially important in a medical setting like Alzheimer's diagnosis, where being able to differentiate accurately between stages (e.g., early mild cognitive impairment vs. late mild cognitive impairment) can significantly impact patient management and treatment planning.

#### CA - 0.521:
Unfortunately, the classification accuracy of our model of 52.1% is relatively low, suggesting that the model is struggling with the complexity of the dataset/nuances in feature interactions that are not linearly separable. In Alzheimer's prediction, where misclassification can have serious consequences (either false positives or false negatives), this low accuracy is problematic, emphasizing the need for more robust models or additional feature engineering such as recursive feature elimination that can handle the complexities of our dataset. This is why we ended up using a neural network as our third model because it is more equipped to handle the complexity of our dataset.

#### F1 Score - 0.452, Precision - 0.470, Recall - 0.521:
With such a low precision, there is a considerable chance this model predicts AD stages incorrectly, which can lead to under or over-treatment, both of which should be avoided. Improving precision might involve refining the feature set to include more definitive diagnostic markers, potentially from genetic profiles or detailed neuropsychological evaluations. The recall is slightly better but still much below what would be acceptable in a medical environment, where identifying just over half of the actual cases correctly is unacceptable. To address these issues, employing techniques like SMOTE (Synthetic Minority Over-sampling Technique) can be particularly effective. SMOTE generates synthetic samples from the minority class (underrepresented stages of AD), thereby balancing the dataset and providing the model with a better opportunity to learn from these rare but critical cases. This balanced dataset can lead to improved recall and a more harmonious balance between precision and recall, thus enhancing the overall F1 score. Such improvements are vital in clinical settings, ensuring that the model not only recognizes more actual cases but also maintains a reasonable level of precision to avoid costly diagnostic errors.

### Neural Network
#### AUC - 0.975:
The AUC of 0.975 demonstrates exceptional model performance in distinguishing between the different classes of Alzheimer's progression in our dataset, from normal cognition to severe Alzheimer's disease. This high AUC is particularly significant in Alzheimer's diagnosis because it indicates that the model has high sensitivity (ability to correctly identify patients with the disease) and specificity (ability to exclude those without the disease). Features such as hippocampal volume and FDG-PET readings, which are known to correlate with Alzheimer’s progression, likely contribute significantly to this high AUC. This reduces both the number of missed diagnoses and (incorrect diagnoses, which are critical in a disease where early detection can substantially alter patient management and outcomes. To further improve AUC, we can refine feature selection based on our PCA to emphasize brain regions most affected early in the disease, such as the entorhinal cortex and fusiform gyrus.

#### CA - 0.863:
With an accuracy of 86.3%, the neural network shows strong capability in correctly identifying the stage of Alzheimer's disease across our dataset with nonlinear and complex patterns. In the clinical context, this means the model can reliably differentiate between stages such as mild cognitive impairment and more severe forms. This accuracy is especially valuable when distinguishing between closely related stages like early mild cognitive impairment (EMCI) and late mild cognitive impairment (LMCI), where clinical interventions might differ significantly. With a relatively high CA to rely on, medical facilities using this model can more reliably plan appropriate interventions, therapies, and support systems tailored to the severity of the disease, thereby potentially slowing its progression and improving the quality of life for patients.

#### F1 Score - 0.862, Precision - 0.862, Recall - 0.863:
The F1 Score for our neural network model shows quality results. This high precision is essential in clinical settings to avoid over-diagnosis, where the implications of a false positive, including creating stress and payment for treatments, are significant.

Given that precision is high, it suggests effective utilization of features such as APOE4 genetic markers and MOCA test scores, which are strongly indicative of Alzheimer’s presence or absence.

Recall of 0.863 indicates that the model successfully identifies 86.3% of all actual Alzheimer’s cases. As mentioned before, high recall is vital for a condition like Alzheimer’s, where early detection can lead to more effective management and potentially delay the progression of the disease.To boost recall, refining the input layer to include more detailed clinical history or expanding the feature set to include additional biomarkers like tau protein levels could be beneficial.

Our high F1 score reflects the model's robustness in handling the trade-offs between missing genuine cases and mistakenly diagnosing healthy individuals.

## Comparison of Algorithm Models

The Naive Bayes model performed well within the context of our project. It particularly excelled in handling high-dimensional datasets, like the one we were working with, offering a straightforward implementation path that scaled well to the size and complexity of our data. Its computational efficiency made it a practical choice for our initial trials, providing a solid baseline upon which we could compare more complex models. However the model’s baseline assumption — that features in the dataset are mutually independent—surfaced as a primary weakness. In the real world, especially in datasets like ours, features often exhibit dependencies that the Naive Bayes model fails to account for. This can lead to unrealistic probability estimates that don't accurately reflect the intricacies of the data, potentially undermining the reliability of its predictions. By oversimplifying the relationships between variables, the model may have overlooked the subtle interactions that are crucial for making accurate predictions needed for Alzheimer’s prediction.

Our logistic regression model performed poorly compared to the other models. The model’s strength was that it was a more robust model than naive bayes with the ability to handle nonlinear relationships between features. It’s also less sensitive to irrelevant features that may not contribute much to the predictions it makes. However, it can be sensitive to outliers in the dataset and multicollinearity between features. A reason for why it might have performed so poorly is that while it can handle nonlinear relationships, the model generally performs better with linear relationships between features which our dataset didn’t have. The model wasn’t able to adequately understand the complex relationships between our features and make accurate predictions.

Our neural network model performed the best overall based on the evaluation statistics. This is because the neural network model’s strengths are well adapted to our dataset, as our dataset has many non-linear and complex relationships that cannot be captured as robustly in naive bayes or logistic regression. This makes sense, because if AD was easy to predict based on certain indicators that linearly pointed to a diagnosis, then AD prediction would be much more widespread. One of the main weaknesses of neural networks is that it is computationally expensive. This model took several hours to run, and due to the black box nature of the model, the reasoning behind predictions can be hard to deduce. 

## Next Steps
Given that our Neural Network performed the best, we should stick with the model and explore ways within the network to further improve accuracies. There are a number of hyperparameters that we can adjust, such as the number of hidden layers, number of neurons per layer, activation functions, learning rates, batch sizes, epochs, etc., to change classification performances. 

One important hyperparameter is deciding how many hidden layers to include in the architecture. Initially, we chose 2 hidden layers with 7 neurons per layer due to us having 7 features and 7 input neurons. However, with further research, it may be of consideration to use 6 neurons per layer instead, since we have 5 neurons in our output layer (5 classes of dementia), as the “rule-of-thumb” is to use less neurons than your input but more than your output. We can also experiment with different numbers of hidden layers and compare classification performances. By adding more parameters, though, we are also risking adding too much complexity and creating an over-fit model. 

Another parameter we can change is which optimizer to use. For our Neural Network, we chose to use stochastic gradient descent as our optimizer. However, different ones exist such as Adam, that may be better at adjusting weights, especially if our hyperparameter initialization was bad. Adjusting the alpha value and max iterations may also improve convergence rates by increasing momentum and decreasing steps to termination. 

Finally, we can also look at testing different activation functions to capture non-linearity. For our model, we used a sigmoid in the hidden layers to help classify patients into the different classes of dementia. Sigmoids will cause neurons to fire based on whether the output was greater than or less than 0.5. This however, may not be enough precision to capture the influence of features in classification of dementia. Functions such as ReLU or hyperbolic tangents (tanh) are worth testing to see if accuracies increase. 

Once parameters are adjusted, we can then test different medical datasets with this algorithm to determine if our model is over-fit for these patients or not. For application into the real-world, it is vital that the diagnosis of an individual with a neurodegenerative disease is unbiased from previous trained data and can be generalized to different communities. While the model can be of use to medical professionals, it is important to consider human input, as the model is clearly not 100% accurate and other characteristics of the individual are worth taking into account. 

# References
Work Cited
[1] Diogo, V.S., Ferreira, H.A., Prata, D. et al. Early diagnosis of Alzheimer’s disease using machine learning: a multi-diagnostic, generalizable approach. Alz Res Therapy 14, 107 (2022). https://doi.org/10.1186/s13195-022-01047-y
[2] Kavitha, C., Mani, V., Srividhya, S. R., Khalaf, O. I., & Tavera Romero, C. A. (2022). Early-Stage Alzheimer's Disease Prediction Using Machine Learning Models. Frontiers in public health, 10, 853294. https://doi.org/10.3389/fpubh.2022.853294
[3] Zhao, Z., Chuah, J. H., Lai, K. W., Chow, C. O., Gochoo, M., Dhanalakshmi, S., Wang, N., Bao, W., & Wu, X. (2023). Conventional machine learning and deep learning in Alzheimer's disease diagnosis using neuroimaging: A review. Frontiers in computational neuroscience, 17, 1038636. https://doi.org/10.3389/fncom.2023.1038636

# Contributions
## Gantt Chart
https://gtvault-my.sharepoint.com/:x:/g/personal/agupta866_gatech_edu/ETpJxKwxeLZIljcBqBqfnC0BvLmJAuWXOKqgEM_MztEdrw?e=2waoQN

## Contribution Table
![alt text](https://github.com/felolivee/AlzheimersDetection/blob/main/contributionTable3.png?raw=true)
