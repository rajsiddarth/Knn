# Implementation of K-Nearest Neighbour Algorithm 

## Dataset
The dataset is provided by a bank with variable descriptions as mentioned below. 

ID: Customer ID,

Age: Customer Age,

Experience: #years of Professional experience,

Income: Annual Income of the customer in $000,

ZIP Code:Home address Zip Code,

Family: Family size of the customer,

CCAvg: Avg spending on credit cards per month in $000,

Education: Education level 1:Undergrad 2: Graduate 3:Advanced/Professional ,

Mortgage: Value of mortgage if any $000,

Securities Account: Does the customer have a securities account with the bank?,

CD Account:Does the customer have a certificate of deposit account with the bank?,

Online : Does the customer use internet banking facilities?,

CreditCard: Does the customer use a credit card issued by the respective bank?,

Personal Loan : Did the customer default on the loan or not ?
 
 ## Project_1
 By using the bank data set we predict the income of the customer based on other independent variables .We estimate the income using KNN 
 regression approach provided by **FNN library** in R.We check for accuracy of our predictions for various k-values(1,3,5,7) and select   the K which gives us the lowest Mean Average Percentage error.
 
 ## Project_2
  The objective of the project is to build a classification model using Knn algorithm to predict whether the customer is going to       default on loan or not. I built the models using sklearn in Python and also libraries in R.The implentations are in R and Python as shown using the default distance measures.
  
 ## Project_3
  Project_3 is based on Project_2 but instead of using the default distance measures, i implemented **value difference measure** as a measure of distance for categorical variables.Value Difference Metric (VDM) is one of the widely used distance metrics for nominal attributes. 
  
  VDF is defined as https://latex.codecogs.com/gif.latex?%5Csum_%7Bh%3D1%7D%5E%7BAll%20classes%7D%20%5Cleft%20%7C%20P%28%5Cfrac%7Bh%7D%7Bval_i%7D%29%29-P%28%5Cfrac%7Bh%7D%7Bval_j%7D%29%20%5Cright%20%7C
  



 

