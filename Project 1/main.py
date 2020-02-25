import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time

pd.set_option('max_columns', 120)
pd.set_option('max_colwidth', 5000)

plt.rcParams['figure.figsize'] = (12, 8)


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

#### DATASET 1 #### 
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


#### DATASET 2 ####
# Predict salary based on attributes dataset

adult_2020 = pd.read_csv('adult.csv', skipinitialspace=True)
adult_2020.columns = ['Age', 'Work class', 'Final Weight', 'Education', 'Education-Num', 'Marital Status', 'Occupation',
                      'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week', 'Native Country',
                      'Salary']
# Dropping Education feature since it is redundant when Education-Num is there
adult_2020 = adult_2020.drop(columns=['Education'])
adult_2020 = adult_2020.drop(columns=['Final Weight'])
adult_2020 = adult_2020.replace('?', np.nan)  # Replace the missing '?' values with Nan to remove them
adult_2020 = adult_2020.dropna(axis=0)
adult_2020 = one_hot_encode(adult_2020, 'Work class')
adult_2020 = one_hot_encode(adult_2020, 'Marital Status')
adult_2020 = one_hot_encode(adult_2020, 'Occupation')
adult_2020 = one_hot_encode(adult_2020, 'Relationship')
adult_2020 = one_hot_encode(adult_2020, 'Race')
adult_2020 = one_hot_encode(adult_2020, 'Sex')
adult_2020 = one_hot_encode(adult_2020, 'Native Country')
adult_2020 = one_hot_encode(adult_2020, 'Salary')
adult_2020 = (adult_2020-adult_2020.min())/(adult_2020.max()-adult_2020.min())  # Regularize the dataset
# null_counts = adult_2020.isnull().sum()
# print(null_counts)


#### DATASET3 ####
abalone_2020 = pd.read_csv('abalone.csv')
abalone_2020.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight',
                        'Shell Weight', 'Rings']
abalone_2020 = abalone_2020.replace('?', np.nan)
a_null = abalone_2020.isnull().sum()
# correlation(abalone_2020, 0.9)
abalone_2020 = one_hot_encode(abalone_2020, 'Sex')
abalone_2020['Rings'][abalone_2020['Rings'] < 10] = 0
abalone_2020['Rings'][abalone_2020['Rings'] >= 10] = 1

#### DATASET 4 ####
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
#####                           DATASET DESCRIPTION                         #####
#################################################################################

# Plot the distribution of a feature for each class
def plot_class_distribution(dataset, feature, binary=True):
  N, D = dataset.shape
  if binary:
    pos = []
    neg = []
    for i in range(N):
      if dataset[i, -1] == 0:
        neg.append(dataset[i, feature])
      else:
        pos.append(dataset[i, feature])
    plt.hist(pos, label="Positives", density=True)
    plt.legend()
    plt.show()
    plt.hist(neg, color="red", label="Negatives", density=True)
    plt.legend()
    plt.show()
  else:
    setosa = []
    versicolor = []
    virginica = []
    for i in range(N):
      if dataset[i, -3] == 1:
        setosa.append(dataset[i, feature])
      elif dataset[i, -2] == 1:
        versicolor.append(dataset[i, feature])
      elif dataset[i, -1] == 1:
        virginica.append(dataset[i, feature])
    plt.hist(setosa, label="Setosa", density=True)
    plt.legend()
    plt.show()
    plt.hist(versicolor, color="red", label="Versicolor", density=True)
    plt.legend()
    plt.show()
    plt.hist(virginica, color="green", label="Virginica", density=True)
    plt.legend()
    plt.show()
    

# Plot the correlation between 2 given features of a dataset
def plot_correlation(dataset, feature1, feature2):
  plt.plot(dataset[:, feature1], dataset[:, feature2], 'o')
  plt.show()



#################################################################################
#####                               CLASSIFIERS                             #####
#################################################################################

class LogisticRegression:
  def __init__(self, lr=0.01, epsilon=1e-2, fit_intercept=True, max_iter=1000):
    self.lr = lr  # Learning Rate
    self.epsilon = epsilon  # Threshold used as a stopping criteria
    self.fit_intercept = fit_intercept
    self.max_iter = max_iter  # Maximum number of iterations
    self.w = np.zeros(1)  # Weights of the model
    self.proba = np.zeros(1)  # Predicted probabilities
    self.labels = np.zeros(1) # Predicted labels

  def sigmoid(self, z):
    h = 1 / (1 + np.exp(-z))
    return h
  
  def cost(self, h, y):
    J = np.mean((-y * np.log(h) - (1-y) * np.log(1 - h)))
    return J

  def gradient(self, X, y, w):
    N, D = X.shape
    z = np.dot(X, self.w)
    yh = self.sigmoid(z)
    grad = np.dot(X.T, yh - y) / N
    return grad
  
  def GradientDescent(self, X, y, lr, epsilon):
    N, D = X.shape
    self.w = np.zeros(D)
    g = np.inf
    k = 0
    while (np.linalg.norm(g) > epsilon) and (k < self.max_iter):
      g = self.gradient(X, y, self.w)
      self.w = self.w - lr*g 

      z = np.dot(X, self.w)
      h = self.sigmoid(z)

      #print(f'loss: {self.cost(h, y)} \t')

      k += 1
    return self.w

  # Compute the parameters of the model
  def fit(self, X, y):
    self.GradientDescent(X, y, self.lr, self.epsilon)
  
  # Compute the probabilities of belonging to each class
  def predict_prob(self, X):
    self.proba = self.sigmoid(np.dot(X, self.w))
    return self.proba

  # Return the predicted labels for each testing sample
  def predict(self, X, threshold=0.5):
    self.predict_prob(X)
    return np.multiply(self.proba >= threshold, 1)


# Gaussian Naive Bayes
class NaiveBayes:
  def __init__(self):
    self.mu = np.zeros(1) # Means of the Gaussians
    self.s = np.zeros(1)  # Standard variation of the Gaussians
    self.log_prior = np.zeros(1)  # Logarithm of the prior probabilities of the classes
    self.proba = np.zeros(1)  # Predicted probabilities
    
  # Compute the parameters of the model
  def fit(self, X, y, noise=0.01):
    # If we are trying to solve a binary classification, create one class for negatives and one class for positives
    if y.ndim == 1:
      N = y.shape
      new_y = (np.vstack([np.ones(N) - y, y])).transpose()
    else:
      new_y = y
    N, C = new_y.shape
    D = X.shape[1]
    # Since some of our features are binary, we add a small noise to the feature matrix so that we do not get a standard deviation of 0
    new_X = X + noise*np.random.rand(N, D)
    self.mu, self.s = np.zeros((C, D)), np.zeros((C, D))
    for c in range(C):
      inds = np.nonzero(new_y[:, c])[0]
      self.mu[c, :] = np.mean(new_X[inds, :], 0)
      self.s[c, :] = np.std(new_X[inds, :], 0)
    self.log_prior = np.log(np.mean(new_y, 0))[:, None]
    return self.mu[: , None, :], self.s[:, None, :], self.log_prior
    
  # Compute the probabilities of belonging to each class
  def predict_prob(self, Xtest):
    log_likelihood = - np.sum(np.log(self.s[:, None, :]) + .5*(((Xtest[None, :, :] - self.mu[: , None, :])/self.s[:, None, :])**2), 2)
    self.proba = self.log_prior + log_likelihood
    return self.proba

  # Return the predicted labels for each testing sample
  def predict(self, Xtest):
    self.predict_prob(Xtest)
    return np.argmax(self.proba, axis=0)
   	


#################################################################################
#####                                METHODS                                #####
#################################################################################

# Count the number of true positives, false positives, true negatives and false negatives
def count_result(target_labels, true_labels):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(target_labels)):
        if target_labels[i] == 0:
            if true_labels[i] == 0:
                tn += 1
            if true_labels[i] == 1:
                fn += 1
        if target_labels[i] == 1:
            if true_labels[i] == 0:
                fp += 1
            if true_labels[i] == 1:
                tp += 1
    return tp, fp, tn, fn

# Define all the different measures that we can use

def accuracy(tp, fp, tn, fn):
    return (tp + tn)/(tp + fp + tn + fn)

def error_rate(tp, fp, tn, fn):
    return (fp + fn)/(tp + fp + tn + fn)

def precision(tp, fp, tn, fn):
    return tp/(tp + fp)

def recall(tp, fp, tn, fn):
    return tp/(tp + fn)

def f1score(tp, fp, tn, fn):
    return 2 * (tp/(tp + fp)) * (tp/(tp + fn)) / ((tp/(tp + fp)) + (tp/(tp + fn)))

# Apply k-fold cross validation on the specified dataset and return the average "obj" measure (accuracy by default)
def cross_validation(X, y, classifier, k=5, obj=accuracy):
  N, D = X.shape
  index = np.linspace(0, N, k+1, False, dtype=np.int)
  total_obj = 0
  for i in range(k):
    classifier.fit(np.vstack([X[:index[i], :], X[index[i+1]:, :]]), np.hstack([y[:index[i]], y[index[i+1]:]]))
    target_labels = classifier.predict(X[index[i]:index[i+1], :])
    tp, fp, tn, fn = count_result(target_labels, y[index[i]:index[i+1]])
    total_obj += obj(tp, fp, tn, fn)
  return total_obj/k

# Evaluate the accuracy
def evaluate_acc(target_labels, true_labels):
  tp, fp, tn, fn = count_result(target_labels, true_labels)
  return accuracy(tp, fp, tn, fn)




#################################################################################
#####                                TRAINING                               #####
#################################################################################

# Find a "good" subset of features for Logistic Regression by optimizing the learning rate, then the epsilon threshold, and finally the maximum number of iteration
# NB : Not a global optimum since we are optimizing each parameter one at a time
# Parameters : pts refer to the number of points, str refer to the starting value, stp refer to the size of the steps
#              lr = Learning Rate ; eps = Epsilon ; iter = Maximun number of Iteration
def find_parameters(training_set, plot=False, obj=accuracy, lr_pts=20, lr_str=0.002, lr_stp=0.002, eps_pts=10, eps_str=0.005, eps_stp=0.005, iter_pts=20, iter_str=200, iter_stp=200):
  # Finding the best learning rate

  lr_lst = []
  acc_lst = []
  best_acc = 0
  best_lr = 0

  for i in range(20):
    logreg = LogisticRegression(lr = 0.002 + 0.002*i)
    acc = cross_validation(training_set[:, :-1], training_set[:, -1], logreg, obj=obj)
    if acc > best_acc:
      best_acc = acc
      best_lr = 0.002 + 0.002*i
    lr_lst.append(0.002 + 0.002*i)
    acc_lst.append(acc)

  print("Best Learning Rate : ", best_lr)

  if plot:
    # Plot the result
    plt.plot(lr_lst, acc_lst)
    plt.suptitle("Accuracy in function of the learning rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.show()


  # Finding the best epsilon

  eps_lst = []
  acc_lst = []
  best_acc = 0
  best_eps = 0

  for i in range(10):
    logreg = LogisticRegression(lr = best_lr, epsilon = 0.005 + 0.005*i)
    acc = cross_validation(training_set[:, :-1], training_set[:, -1], logreg)
    if acc > best_acc:
      best_acc = acc
      best_eps = 0.005 + 0.005*i
    eps_lst.append(0.005 + 0.005*i)
    acc_lst.append(acc)

  print("Best Epsilon : ", best_eps)

  if plot:
    # Plot the result
    plt.plot(eps_lst, acc_lst)
    plt.suptitle("Accuracy in function of the epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()


  # Finding the best number of iteration

  iter_lst = []
  acc_lst = []
  best_acc = 0
  best_iter = 0

  for i in range(20):
    logreg = LogisticRegression(lr = best_lr, epsilon = best_eps, max_iter = 200 + 200*i)
    acc = cross_validation(training_set[:, :-1], training_set[:, -1], logreg)
    if acc > best_acc:
      best_acc = acc
      best_iter = 200 + 200*i
    iter_lst.append(200 + 200*i)
    acc_lst.append(acc)

  print("Best Number of Iteration : ", best_iter)

  if plot:
    # Plot the result
    plt.plot(iter_lst, acc_lst)
    plt.suptitle("Accuracy in function of the number of iteration")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.show()

  return best_lr, best_eps, best_iter, best_acc


#################################################################################
#####                              IRIS DATASET                             #####
#################################################################################

# We define specific methods for the Iris dataset because in this case, we wish to classify our data into 3 classes (not a binary classification)

# Compute the accuracy of the result
def iris_acc(predicted_class, true_class):
  correct = 0
  total = 0
  for i in predicted_class:
    if (true_class[i, predicted_class[i]] == 1):
      correct += 1
    total += 1
  return correct / total

# Predict the classes using Logistic Regression
def iris_logreg_predict(training_set, testing_set, lr=0.01, eps=0.01, max_iter=1000):
  # Train a specific classifier for each of the 3 classes
  setosa_logreg = LogisticRegression(lr, eps, max_iter=max_iter)
  versicolor_logreg = LogisticRegression(lr, eps, max_iter=max_iter)
  virginica_logreg = LogisticRegression(lr, eps, max_iter=max_iter)

  setosa_logreg.fit(training_set[:, :-3], training_set[:, -3])
  versicolor_logreg.fit(training_set[:, :-3], training_set[:, -2])
  virginica_logreg.fit(training_set[:, :-3], training_set[:, -1])

  # Compute the probabilities of belonging to each class
  proba = setosa_logreg.predict_prob(testing_set[:, :-3])
  proba = np.vstack([proba, versicolor_logreg.predict_prob(testing_set[:, :-3])])
  proba = np.vstack([proba, virginica_logreg.predict_prob(testing_set[:, :-3])])

  # The final predicted class will be the one with the highest probability
  iris_class = np.argmax(proba, axis=0)

  return iris_acc(iris_class, testing_set[:, -3:])

# Apply k-fold cross validation with Logistic Regression
def iris_logreg_cv(dataset, lr=0.01, eps=0.01, max_iter=1000, k=5, shuffle=False):
  if shuffle:
    np.random.shuffle(dataset)
  N, D = dataset.shape
  index = np.linspace(0, N, k+1, False, dtype=np.int)
  total_acc = 0
  for i in range(k):
    # Split the dataset
    training_set = np.vstack([dataset[:index[i], :], dataset[index[i+1]:, :]])
    testing_set = dataset[index[i]:index[i+1], :]
    # Train and test the model
    total_acc += iris_logreg_predict(training_set, testing_set, lr, eps, max_iter)
  return total_acc/k

# Find a "good" subset of features for Logistic Regression by optimizing the learning rate, then the epsilon threshold, and finally the maximum number of iteration
# NB : Not a global optimum since we are optimizing each parameter one at a time
# Parameters : pts refer to the number of points, str refer to the starting value, stp refer to the size of the steps
#              lr = Learning Rate ; eps = Epsilon ; iter = Maximun number of Iteration
def find_parameters_iris(training_set, plot=False, lr_pts=20, lr_str=0.002, lr_stp=0.002, eps_pts=10, eps_str=0.005, eps_stp=0.005, iter_pts=20, iter_str=200, iter_stp=200):
  # Finding the best learning rate

  lr_lst = []
  acc_lst = []
  best_acc = 0
  best_lr = 0

  for i in range(20):
    acc = iris_logreg_cv(training_set, lr=0.002 + 0.002*i)
    if acc > best_acc:
      best_acc = acc
      best_lr = 0.002 + 0.002*i
    lr_lst.append(0.002 + 0.002*i)
    acc_lst.append(acc)

  print("Best Learning Rate : ", best_lr)

  if plot:
    # Plot the result
    plt.plot(lr_lst, acc_lst)
    plt.suptitle("Accuracy in function of the learning rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.show()


  # Finding the best epsilon

  eps_lst = []
  acc_lst = []
  best_acc = 0
  best_eps = 0

  for i in range(10):
    acc = iris_logreg_cv(training_set, lr=best_lr, eps=0.005 + 0.005*i)
    if acc > best_acc:
      best_acc = acc
      best_eps = 0.005 + 0.005*i
    eps_lst.append(0.005 + 0.005*i)
    acc_lst.append(acc)

  print("Best Epsilon : ", best_eps)

  if plot:
    # Plot the result
    plt.plot(eps_lst, acc_lst)
    plt.suptitle("Accuracy in function of the epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()


  # Finding the best number of iteration

  iter_lst = []
  acc_lst = []
  best_acc = 0
  best_iter = 0

  for i in range(20):
    acc = iris_logreg_cv(training_set, lr=best_lr, eps=best_eps, max_iter=200+200*i)
    if acc > best_acc:
      best_acc = acc
      best_iter = 200 + 200*i
    iter_lst.append(200 + 200*i)
    acc_lst.append(acc)

  print("Best Number of Iteration : ", best_iter)

  if plot:
    # Plot the result
    plt.plot(iter_lst, acc_lst)
    plt.suptitle("Accuracy in function of the number of iteration")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.show()

  return best_lr, best_eps, best_iter, best_acc

# Return a graph that show the evolution of the "obj" measure for Logistic Regression depending on the percentage of the dataset used for training 
def iris_evaluate_logreg(dataset):
  perc_lst = []
  acc_lst = []
  train_lst = []
  N, D = dataset.shape
  for i in range(7):
    # Shuffle and Split the dataset
    np.random.shuffle(dataset)
    train_percent = 0.3 + 0.1*i

    train_size = math.floor(N*train_percent)
    training_set = dataset[:train_size, :]
    testing_set = dataset[train_size:, :]

    # Find good parameters with cross-validation
    lr, eps, iter, train_acc = find_parameters_iris(training_set)
    train_lst.append(train_acc)

    # Final training and testing
    acc = iris_logreg_predict(training_set, testing_set, lr, eps, iter)
    # Report the results
    perc_lst.append(0.1 + 0.1*i)
    acc_lst.append(acc)

  # Plot the results
  plt.plot(perc_lst, acc_lst, label="Testing Accuracy")
  plt.plot(perc_lst, train_lst, color="red", label="Training Accuracy")
  plt.suptitle("Accuracy in function of the training percentage")
  plt.xlabel("Training Percentage")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()


# Apply k-fold cross validation with Naive Bayes
def iris_nb_cv(dataset, k=5, shuffle=False):
  if shuffle:
    np.random.shuffle(dataset)
  N, D = dataset.shape
  index = np.linspace(0, N, k+1, False, dtype=np.int)
  nb = NaiveBayes()
  total_acc = 0
  for i in range(k):
    # Split the dataset
    training_set = np.vstack([dataset[:index[i], :], dataset[index[i+1]:, :]])
    testing_set = dataset[index[i]:index[i+1], :]

    # Train and test the model
    nb.fit(training_set[:, :-3], training_set[:, -3:])
    predicted_class = nb.predict(testing_set[:, :-3])

    total_acc += iris_acc(predicted_class, testing_set[:, -3:])
  return total_acc/k

# Return a graph that show the evolution of the "obj" measure for Naive Bayes depending on the percentage of the dataset used for training
def iris_evaluate_nb(dataset):
  perc_lst = []
  acc_lst = []
  train_lst = []
  N, D = dataset.shape
  for i in range(7):
    # Shuffle and Split the dataset
    np.random.shuffle(dataset)
    train_percent = 0.3 + 0.1*i

    train_size = math.floor(N*train_percent)
    training_set = dataset[:train_size, :]
    testing_set = dataset[train_size:, :]

    # Apply cross-validation
    train_acc = iris_nb_cv(training_set)
    train_lst.append(train_acc)

    # Final Training and Testing
    nb = NaiveBayes()
    nb.fit(training_set[:, :-3], training_set[:, -3:])
    predicted_class = nb.predict(testing_set[:, :-3])

    acc = iris_acc(predicted_class, testing_set[:, -3:])

    # Report the results
    perc_lst.append(0.1 + 0.1*i)
    acc_lst.append(acc)

  # Plot the results
  plt.plot(perc_lst, acc_lst, label="Testing Accuracy")
  plt.plot(perc_lst, train_lst, color="red", label="Training Accuracy")
  plt.suptitle("Accuracy in function of the training percentage")
  plt.xlabel("Training Percentage")
  plt.ylabel("Accuracy")
  plt.show()


# Return a graph comparing the accuracy (or the specified measure "obj") of Logistic Regression and Naive Bayes
def iris_compare(dataset, train_percent=0.8):
  # Shuffle and Split the dataset
  np.random.shuffle(dataset)
  N, D = dataset.shape
  
  train_size = math.floor(N*train_percent)
  training_set = dataset[:train_size, :]
  testing_set = dataset[train_size:, :]

  # Find good parameters for Logistic Regression using cross-validation
  lr, eps, iter, lr_train_acc = find_parameters_iris(training_set)
  # Final Training and Testing using Logistic Regression
  lr_acc = iris_logreg_predict(training_set, testing_set, lr, eps, iter)

  # Apply cross-validation with Naive Bayes
  nb = NaiveBayes()
  nb_train_acc = iris_nb_cv(training_set)

  # Final Training and Testing using Naive Bayes
  nb.fit(training_set[:, :-3], training_set[:, -3:])
  predicted_class = nb.predict(testing_set[:, :-3])
  nb_acc = iris_acc(predicted_class, testing_set[:, -3:])

  # Plot the result
  labels = ["Training Accuracy", "Testing Accuracy"]
  x = np.arange(2)
  width = 0.35

  fig, ax = plt.subplots()
  ax.bar(x - width/2, [lr_train_acc, lr_acc], width, label='Logistic Regression')
  ax.bar(x + width/2,[nb_train_acc, nb_acc], width, label='Naive Bayes')

  ax.set_ylabel('Accuracy')
  ax.set_title('Comparison between Logistic Regression and Naive Bayes')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()

  plt.show()

  return(lr_train_acc, nb_train_acc, lr_acc, nb_acc)



#################################################################################
#####                             RESULT ANALYSIS                           #####
#################################################################################


# Return a graph comparing the accuracy (or the specified measure "obj") of Logistic Regression and Naive Bayes
def compare(dataset, train_percent=0.8, obj=accuracy):
  # Shuffle and split the dataset
  np.random.shuffle(dataset)
  N, D = dataset.shape

  train_size = math.floor(N*train_percent)
  training_set = dataset[:train_size, :]
  testing_set = dataset[train_size:, :]

  lr = LogisticRegression()
  nb = NaiveBayes()
  
  # Use cross-validation on the training set
  train_acc_lr = cross_validation(training_set[:, :-1], training_set[:, -1], lr, obj=obj)
  train_acc_nb = cross_validation(training_set[:, :-1], training_set[:, -1], nb, obj=obj)

  # Final training and testing

  lr.fit(training_set[:, :-1], training_set[:, -1])
  lr_class = lr.predict(testing_set[:, :-1])
  tp, fp, tn, fn = count_result(lr_class, testing_set[:, -1])
  acc_lr = obj(tp, fp, tn, fn)

  nb.fit(training_set[:, :-1], training_set[:, -1])
  nb_class = nb.predict(testing_set[:, :-1])
  tp, fp, tn, fn = count_result(nb_class, testing_set[:, -1])
  acc_nb = accuracy(tp, fp, tn, fn)

  # Plot the results

  labels = ["Training Accuracy", "Testing Accuracy"]
  x = np.arange(2)
  width = 0.35

  fig, ax = plt.subplots()
  ax.bar(x - width/2, [train_acc_lr, acc_lr], width, label='Logistic Regression')
  ax.bar(x + width/2,[train_acc_nb, acc_nb], width, label='Naive Bayes')

  ax.set_ylabel('Accuracy')
  ax.set_title('Comparison between Logistic Regression and Naive Bayes')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()

  plt.show()

  return(train_acc_lr, train_acc_nb, acc_lr, acc_nb)

# Report the different measures for Logistic Regression after having found good parameters
def evaluate_log(dataset, train_percent=0.8, plot=False, obj=accuracy):
  # Shuffle and Split the dataset
  np.random.shuffle(dataset)
  N, D = dataset.shape

  train_size = math.floor(N*train_percent)
  training_set = dataset[:train_size, :]
  testing_set = dataset[train_size:, :]

  # Find good parameters using cross-validation
  lr, eps, iter, train_acc = find_parameters(training_set, plot=plot, obj=obj)

  # Final Training and Testing
  logreg = LogisticRegression(lr, eps, max_iter=iter)
  logreg.fit(training_set[:, :-1], training_set[:, -1])
  target_labels = logreg.predict(testing_set[:, :-1])
  tp, fp, tn, fn = count_result(target_labels, testing_set[:, -1])

  # Report the different measures
  print("Accuracy : ", accuracy(tp, fp, tn, fn))
  print("Error Rate : ", error_rate(tp, fp, tn, fn))
  print("Precision : ", precision(tp, fp, tn, fn))
  print("Recall : ", recall(tp, fp, tn, fn))
  print("F1-score : ", f1score(tp, fp, tn, fn))

  return train_acc, obj(tp, fp, tn, fn)

# Return a graph that show the evolution of the "obj" measure for Logistic Regression depending on the percentage of the dataset used for training
def evaluate_model(dataset):
  perc_lst = []
  acc_lst = []
  train_lst = []
  for i in range(9):
    # Train and test the model, then report the results
    train_acc, acc = evaluate_log(dataset, train_percent = 0.1 + 0.1*i)
    perc_lst.append(0.1 + 0.1*i)
    acc_lst.append(acc)
    train_lst.append(train_acc)

  # Plot the result
  plt.plot(perc_lst, acc_lst, label="Testing Accuracy")
  plt.plot(perc_lst, train_lst, color="red", label="Training Accuracy")
  plt.suptitle("Accuracy in function of the training percentage")
  plt.xlabel("Training Percentage")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()

# Return a graph that show the evolution of the "obj" measure for Logistic Regression depending on the percentage of the dataset used for training
def evaluate_nb(dataset, obj=accuracy):
  perc_lst = []
  acc_lst = []
  train_lst = []
  for i in range(9):
    # Shuffle and split the dataset
    train_percent = 0.1 + 0.1*i
    np.random.shuffle(dataset)
    N, D = dataset.shape

    train_size = math.floor(N*train_percent)
    training_set = dataset[:train_size, :]
    testing_set = dataset[train_size:, :]

    nb = NaiveBayes()

    # Train and test the model
    train_acc = cross_validation(training_set[:, :-1], training_set[:, -1], nb, obj=obj)
    nb.fit(training_set[:, :-1], training_set[:, -1])
    target_labels = nb.predict(testing_set[:, :-1])

    # Compute the accuracy (or the specified measure "obj")
    tp, fp, tn, fn = count_result(target_labels, testing_set[:, -1])
    acc = obj(tp, fp, tn, fn)

    # Add the result
    perc_lst.append(train_percent)
    acc_lst.append(acc)
    train_lst.append(train_acc)

  # Plot the result
  plt.plot(perc_lst, acc_lst, label="Testing Accuracy")
  plt.plot(perc_lst, train_lst, color="red", label="Training Accuracy")
  plt.suptitle("Accuracy in function of the training percentage")
  plt.xlabel("Training Percentage")
  plt.ylabel("Accuracy")
  plt.show()


dataset = iris_array

train_percent = 0.8
lr = LogisticRegression()
nb = NaiveBayes()
np.random.shuffle(dataset)
N, D = dataset.shape

train_size = math.floor(N*train_percent)
training_set = dataset[:train_size, :]
testing_set = dataset[train_size:, :]

t1 = time.time()
#cross_validation(training_set[:, :-1], training_set[:, -1], lr)
iris_logreg_cv(dataset)
t2 = time.time()
#cross_validation(training_set[:, :-1], training_set[:, -1], nb)
iris_nb_cv(dataset)
t3 = time.time()

print(t2-t1, t3-t2)

"""
lr : 0.3739914894104004 - nb : 0.0029909610748291016  ; lr : 0.36476588249206543 - nb : 0.003989219665527344
lr : 37.25168251991272 - nb : 0.9193809032440186  ; lr : 30.400632619857788 - nb : 1.3094234466552734
lr : 1.6502840518951416 - nb : 0.029920339584350586 ; lr : 1.4717061519622803 - nb : 0.03091740608215332
lr : 0.6672160625457764 - nb : 0.002992868423461914 ; lr : 0.8187668323516846 - nb : 0.001993894577026367
"""

"""
# Uncomment the dataset that you want to use
dataset = ionosphere_array
#dataset = adult_array
#dataset = abalone_array

# Uncomment to get graph comparing accuracy between Logistic Regression and Naive Bayes
compare(dataset)
# Uncomment to get graphs about accuracy depending on the learning rate, epsilon and the number of iteration
# Parameters : obj = evaluation function used; lr_pts = number of points computed for learning rate; lr_str = starting number for learning rate; lr_stp = size of the steps for learning rate; etc...
evaluate_log(dataset, plot=True)
# Uncomment to get graphs about accuracy of logistic refression depending on the percentage of the dataset used for training
# Parameters : obj = evaluation function used;
evaluate_model(dataset)

# Uncomment to get graphs about accuracy of logistic refression depending on the percentage of the dataset used for training
evaluate_nb(dataset)


### Evaluation about the Iris dataset (classification into 3 classes)

#dataset = iris_array

#iris_compare(dataset)
# Uncomment to get graphs about accuracy depending on the learning rate, epsilon and the number of iteration
#find_parameters_iris(dataset, plot=True)
# Uncomment to get graphs about accuracy depending on the percentage of the dataset used for training
#iris_evaluate_logreg(dataset)
#iris_evaluate_nb(dataset)
"""



