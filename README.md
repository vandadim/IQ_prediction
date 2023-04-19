# IQ prediction

# Intelligence, brain size, and brain morphometry: resolving conflicting claims


Brain size is associated with intelligence, but the relationship between brain morphometric measures and intelligence is unclear. Studies have produced conflicting results or no significant relations between intelligence and morphometric measures such as cortical thickness and peri-cortical contrast. This discrepancy may be due to a failure to fully account for the relationship between brain size and intelligence. Our study shows that neither cortical thickness nor peri-cortical contrast improves IQ prediction accuracy beyond what is achieved with brain volume alone.

**TABLE OF CONTENTS**
===================================
- [Introduction](#introduction)
- [Dataset](#dataset)
  - [ABCD](#dataset)
  - [NIHPD](#dataset)
  - [NKI-RS](#dataset)
- [Methods](#methods)
  - [Preprocessing](#methods)
  - [Regression Models](#methods)
    - [Elastic Net](#methods)
    - [Random Forest](#methods)
    - [SVR](#methods)
    - [XGBoost](#methods)
- [Results](#results)
  - [ABCD Results](#results)
  - [NIHPD Results](#results)
  - [NKI-RS Results](#results)


## Introduction

Our study aimed to evaluate the impact of morphometric measures on predicting IQ scores. To do this, we first predicted IQ scores using brain volume, age, sex, and scanner manufacturer as variables. We then repeated this process while adding either cortical thickness or peri-cortical contrast as additional variables. By comparing the accuracy of these two sets of predictions, we could determine the extent to which the morphometric measures improved the accuracy of IQ score predictions.

It's important to note that we did not adjust for brain size or eliminate its influence. Instead, we sought to determine whether including morphometric measures provided additional information that improved the accuracy of IQ score predictions.


## Dataset

This work presents three distinct studies that investigate the prediction of IQ scores across different age groups. Firstly, we assess IQ prediction in children using data from the first visit of the ABCD study. Secondly, we explore IQ prediction in developmental data, specifically in subjects ranging from middle childhood through adolescence, using data from the first visit of the NIH pediatric dataset. Finally, we investigate IQ prediction in adults, utilizing data from the Nathan-Klein Institute - Rockland Sample.
The roster identification numbers (RIDs) of the subjects employed in this study are provided in the following links:
- [ABCD](/Subjects%20RIDs/ABCD%20Data/)
- [NIHPD](/Subjects%20RIDs/NIHPD%20Data/)
- [NKI-RS](/Subjects%20RIDs/NKI-RS%20Data/)
## Methods

### Preprocessing
For more comprehensive information about the preprocessing techniques utilized in our study, please refer to the following link provided in our paper: WWW.YYYY.CCC

### Regression Models
We employed four different regression models in our study: elastic net (implemented via the Glmnet package), support vector regression (SVR), random forest, and XGBoost. Here's a brief overview of each method:

- [Elastic net (Glmnet)]: This method is a regularized regression technique that combines both L1 and L2 penalties to improve the stability and interpretability of the model. In our study, we utilized the implementation of elastic net provided in the Glmnet package. To access the code used for elastic net in our study, please click on the following link: [Elastic Net](/Regression%20Models/Elastic%20Net/).
- [Random forest (RF)]: Random forest is an ensemble learning technique that utilizes multiple decision trees to generate a final prediction. This method is known for its ability to handle complex data and avoid overfitting. To access the code for random forest implementation in our study, please click on the following link: [Random Forest](/Regression%20Models/Random%20Forest/).

- [Support vector regression (SVR)]: SVR is a powerful machine learning technique that utilizes support vector machines to perform regression analysis. In our study, we utilized the implementation of SVR provided in the scikit-learn library. To access the code used for SVR in our study, please click on the following link: [SVR](/Regression%20Models/SVR/).
- [XGBoost]: XGBoost is an optimized implementation of gradient boosting that is widely used in machine learning competitions due to its high predictive accuracy. For our study, we employed the XGBoost implementation provided in the xgboost library. To access the code used for XGBoost in our study, please refer to the following link: [XGBoost](/Regression%20Models/XGBoost/).

## Results

### The summarized results for each dataset in this study are provided in the following links:
- [ABCD Results](/Results/ABCD%20Data/)
- [NIHPD Results](/Results/NIHPD%20Data/)
- [NKI-RS Results](/Results/NKI-RS%20Data/)


