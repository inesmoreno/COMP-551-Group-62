import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('max_columns', 120)
pd.set_option('max_colwidth', 5000)

plt.rcParams['figure.figsize'] = (12, 8)




#################################################################################
#####                               CLASSIFIERS                             #####
#################################################################################

### Logistic Regression
class logreg:
    def __init__(self):

    def fit(X, y):

    def predict(X):

### Naive Bayes
class NB:
    def __init__(self):

    def fit(X, y):

    def predict(X):

#################################################################################
#####                            PRE-PROCESSING                             #####
#################################################################################

# One hot encoding for selected categorical features, removes the original non-encoded feature and concatenates
# the original dataframe to the new one hot encoded feature(s)

def one_hot_encode(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    result = pd.concat([original_dataframe, dummies], axis=1)
    result = result.drop([feature_to_encode], axis=1)
    return result


# Function that deletes columns that are above a certain threshold of correlation of our choosing
# Depending on the dataset characteristics

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]

    # print(dataset)


# Predict whether radar result is "good"/"bad" dataset
ionosphere_2020 = pd.read_csv('ionosphere.csv')
ionosphere_2020.columns = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7',
                           'Feature 8', 'Feature 9', 'Feature 10', 'Feature 11', 'Feature 12', 'Feature 13',
                           'Feature 14', 'Feature 15', 'Feature 16', 'Feature 17', 'Feature 18', 'Feature 19',
                           'Feature 20', 'Feature 21', 'Feature 22', 'Feature 23', 'Feature 24', 'Feature 25',
                           'Feature 26', 'Feature 27', 'Feature 28', 'Feature 29', 'Feature 30', 'Feature 31',
                           'Feature 32', 'Feature 33', 'Feature 34', 'Result']
# Dropping Feature 2 as it only has zeros
ionosphere_2020 = ionosphere_2020.drop(columns=['Feature 2'])
ionosphere_2020 = one_hot_encode(ionosphere_2020, 'Result')
correlation(ionosphere_2020, 0.98)
# print(ionosphere_2020.var(axis=0))
# print(ionosphere_2020.nunique())
# null_counts2 = ionosphere_2020.isnull().sum()
# print(null_counts2)


# Predict salary based on attributes dataset

adult_2020 = pd.read_csv('adult.csv', skipinitialspace=True)
adult_2020.columns = ['Age', 'Work class', 'Final Weight', 'Education', 'Education-Num', 'Marital Status', 'Occupation',
                      'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week', 'Native Country',
                      'Salary']
# Dropping Education feature since it is redundant when Education-Num is there
adult_2020 = adult_2020.drop(columns=['Education'])
adult_2020 = adult_2020.replace('?', np.nan)  # Replace the missing '?' values with Nan to remove them
adult_2020 = adult_2020.dropna(axis=0)
adult_2020 = one_hot_encode(adult_2020, 'Work class')
adult_2020 = one_hot_encode(adult_2020, 'Marital Status')
adult_2020 = one_hot_encode(adult_2020, 'Relationship')
adult_2020 = one_hot_encode(adult_2020, 'Race')
adult_2020 = one_hot_encode(adult_2020, 'Sex')
adult_2020 = one_hot_encode(adult_2020, 'Native Country')
adult_2020 = one_hot_encode(adult_2020, 'Salary')
# null_counts = adult_2020.isnull().sum()
# print(null_counts)

abalone_2020 = pd.read_csv('abalone.csv')
abalone_2020.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight',
                        'Shell Weight', 'Rings']
abalone_2020 = abalone_2020.replace('?', np.nan)
a_null = abalone_2020.isnull().sum()
# correlation(abalone_2020, 0.9)
abalone_2020 = one_hot_encode(abalone_2020, 'Sex')

iris_2020 = pd.read_csv('iris.csv')
iris_2020.columns = ['Sepal length', 'Sepal Width', 'Petal length', 'Petal width', 'Class']
iris_2020 = iris_2020.replace('?', np.nan)
iris_2020 = one_hot_encode(iris_2020, 'Class')
# print(iris_2020.var(axis=0))

adult_array = adult_2020.to_numpy()
ionosphere_array = ionosphere_2020.to_numpy()
abalone_array = abalone_2020.to_numpy()
iris_array = iris_2020.to_numpy()

#################################################################################
#####                          DATASET DESCRIPTION                          #####
#################################################################################



#################################################################################
#####                                TRAINING                               #####
#################################################################################



#################################################################################
#####                             RESULT ANALYSIS                           #####
#################################################################################
