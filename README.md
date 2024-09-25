# Wine Quality Prediction
## Dataset
The dataset is public on Kaggle with about 6498 entries. They records the data with 11 dimentions including: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, 
total sulfur dioxide, density, pH, sulphates, and alcohol. With these data we want to predict the wine quality ranging from 1 to 10 (10 is the best). 

## Data preprocessing 
First we need to check all null data in the file, using ''' print(data.isnull().sum()) 
## Data visualization 
First the histogram for each dimenstion was analyzed.
![histogram](https://github.com/duhanlu/wineQuality/blob/main/his_pic.png)
Then the correlation between dimensions and also with the quality is analyzed as well. 

![correlation](https://github.com/duhanlu/wineQuality/blob/main/correlation.png)
## Model choise
This is a claasification problem with multi-classes. The summary of classification algorithm is as below: 
- Logistic Regression
- Support Vector Machine (used for binary classification but can be used as multi-classification by treating two class with one class and other classes)
- Decision Trees: used for small size of dataset 
- Random Forest: used for a large size of dataset 
- nearest neighbor 
- naive bayes
- neural network

I've tried these classificatin algorithm to compare the pereformance. Following is the result for each algorithm. 
![correlation](https://github.com/duhanlu/wineQuality/blob/main/correlation.png)

  

## Result 
https://github.com/duhanlu/wineQuality/blob/main/his_pic.png
