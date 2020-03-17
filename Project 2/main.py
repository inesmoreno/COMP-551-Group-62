import numpy as np
from glob import glob
import nltk
import scipy
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
import sklearn as sk
import sklearn.naive_bayes as nb
import sklearn.tree as sktree
import sklearn.ensemble as skens
import math
import matplotlib.pyplot as plt
import os, re, string
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV

# Get the data
from sklearn.datasets import fetch_20newsgroups

#nltk.download()


##########################################################
### Transforming the text into vectors + preprocessing ###
##########################################################

### Feature extraction : transform each line into a vector of features
def get_vectors(vectorizer, raw_X, frequency=True, downscale=True, fit=True):
    # Create a vector of the number of occurence of each word
    if fit:
        X = vectorizer.fit_transform(raw_X)
    else:
        X = vectorizer.transform(raw_X)
    if frequency:
        if downscale:
            X = sk.feature_extraction.text.TfidfTransformer(use_idf=True).fit_transform(X)      # Change the vector of occurence into a vector of frequencies, with high frequencies being downscaled
        else:
            X = sk.feature_extraction.text.TfidfTransformer(use_idf=False).fit_transform(X)     # Change the vector of occurence into a vector of frequencies
    return X


#######################################################################################
### Using the model to train on a training set then predict result on a testing set ###
#######################################################################################

### Test a classifier : train the classifier on the training set and use it to predict the results on the testing set
def test_classifier(clf, train_set, test_set):
    clf.fit(train_set[:, :-1], np.ravel(train_set[:, -1].toarray()))        # Train on the training set
    res = clf.predict(test_set[:, :-1])     # Predict the result on the testing set
    return res


##################################################
### Evaluate the performance of the classifier ###
##################################################

# Count the number of true positive, true negative, false positive and false negative
def confusion_matrix(predicted, labels, sparse=True):
    if sparse:
        correct = np.ravel(labels.toarray())
    else:
        correct = labels
    classes = np.unique(correct)        # List the different labels found in the testing set
    print(classes)
    conf_mat = np.zeros((len(classes), len(classes)))       # Count the number of sample that was predicted to be labelled i with actual label j
    for i in range(len(predicted)):
        pred = np.argwhere(classes == predicted[i])[0]
        corr = np.argwhere(classes == correct[i])[0]
        conf_mat[pred[0], corr[0]] += 1
    return conf_mat

# Compute the accuracy of the model
def accuracy(matrix):
    n = matrix.shape[0]
    correct = 0     # Count the number of correctly predicted samples
    for i in range(n):
        correct += matrix[i, i]
    return correct/np.sum(matrix)

# Compute the log cross entropy of the model
# Note that this measure might be misleading if every sample labelled j is wrongly predicted to be labelled with the same label i
def log_entropy(matrix):
    n = matrix.shape[0]
    H = 0
    total = np.sum(matrix)
    predicted_classes = np.sum(matrix, axis=1)
    for i in range(n):
        partial_H = 0
        if predicted_classes[i] != 0:
            for j in range(n):
                if matrix[i, j] != 0:
                    partial_H += -(matrix[i, j]/predicted_classes[i])*math.log(matrix[i, j]/predicted_classes[i])
            H += (predicted_classes[i]/total)*partial_H
    return H

# Return the accuracy and the log cross entropy of the model
def evaluate(matrix, display=False):
    acc = accuracy(matrix)
    H = log_entropy(matrix)
    if display:
        print('Accuracy : ', acc)
        print('Log Cross Entropy : ', H)
    return acc, H



#################################################
### Fit the parameters using the training set ###
#################################################

# Return the average accuracy and log cross entropy of the model after applying k-fold cross validation
def cross_validation(clf, dataset, k=5):
    indexes = sk.model_selection.KFold(n_splits=k).split(dataset)       # Get the indexes that we will use for the k-fold cross validation
    avg_acc = 0
    avg_ent = 0
    for train, test in indexes:
        # Separate the dataset into a training set and a testing set
        train_set = dataset[train]
        test_set = dataset[test]
        # Predict the result and evaluate the performance
        res = test_classifier(clf, train_set, test_set)
        mat = confusion_matrix(res, test_set[:, -1])
        acc, ent = evaluate(mat)
        avg_acc += acc
        avg_ent += ent
    return avg_acc/k, avg_ent/k


###################################################################################
### Analysing the effect of preprocessing on the performance of the classifiers ###
###################################################################################

def best_ngram(min_n, max_n, vectorizer, clf, raw_X, Y, k=5):
    acc_list, ent_list = [], []
    ngram_lst = []
    sparse_Y = scipy.sparse.csr_matrix(Y).transpose()
    for i in range(min_n, (max_n+1)):
        for j in range(i, (max_n+1)):
            vectorizer.set_params(ngram_range=(i, j))
            ngram_lst.append('(' + str(i) + ', ' + str(j) + ')')
            X = get_vectors(vectorizer, raw_X)
            dataset = scipy.sparse.hstack([X, sparse_Y], format="csr")
            dataset = sk.utils.shuffle(dataset)
            acc, ent = cross_validation(clf, dataset, k)
            acc_list.append(acc)
            ent_list.append(ent)

    plt.bar(ngram_lst, acc_list)
    plt.ylabel('Accuracy')
    plt.title('Accuracy in function of the ngram used')
    plt.show()
    plt.bar(ngram_lst, ent_list)
    plt.ylabel('Log Cross Entropy')
    plt.title('Log Cross Entropy in function of the ngram used')
    plt.show()

### For a given classifier, compute the perfomance metrics depending on the stopwords list 
def test_tokenizer(token_list, names_list, vectorizer, clf, raw_X, Y, k=5):
    acc_list, ent_list = [], []
    sparse_Y = scipy.sparse.csr_matrix(Y).transpose()
    for t in token_list:
        vectorizer.set_params(tokenizer=t)
        X = get_vectors(vectorizer, raw_X)
        dataset = scipy.sparse.hstack([X, sparse_Y], format="csr")
        dataset = sk.utils.shuffle(dataset)
        acc, ent = cross_validation(clf, dataset, k)
        acc_list.append(acc)
        ent_list.append(ent)

    plt.bar(names_list, acc_list)
    plt.ylabel('Accuracy')
    plt.title('Accuracy in function of the tokenizer used')
    plt.show()
    plt.bar(names_list, ent_list)
    plt.ylabel('Log Cross Entropy')
    plt.title('Log Cross Entropy in function of the tokenizer used')
    plt.show()

### For a given classifier, compare the performance depending on the method used to count the words
def test_counting(vectorizer, clf, raw_X, Y, k=5):
    names_list = ['Occurence', 'Frequency', 'Downscaled frequency']
    acc_list, ent_list = [], []
    sparse_Y = scipy.sparse.csr_matrix(Y).transpose()
    
    # Occurence counting
    X = get_vectors(vectorizer, raw_X, frequency=False, downscale=False)
    dataset = scipy.sparse.hstack([X, sparse_Y], format="csr")
    dataset = sk.utils.shuffle(dataset)
    acc, ent = cross_validation(clf, dataset, k)
    acc_list.append(acc)
    ent_list.append(ent)

    # Frequency counting
    X = get_vectors(vectorizer, raw_X, downscale=False)
    dataset = scipy.sparse.hstack([X, sparse_Y], format="csr")
    dataset = sk.utils.shuffle(dataset)
    acc, ent = cross_validation(clf, dataset, k)
    acc_list.append(acc)
    ent_list.append(ent)

    # Downscaled frequency counting
    X = get_vectors(vectorizer, raw_X)
    dataset = scipy.sparse.hstack([X, sparse_Y], format="csr")
    dataset = sk.utils.shuffle(dataset)
    acc, ent = cross_validation(clf, dataset, k)
    acc_list.append(acc)
    ent_list.append(ent)


    plt.bar(names_list, acc_list)
    plt.ylabel('Accuracy')
    plt.title('Accuracy in function of the counting method')
    plt.show()
    plt.bar(names_list, ent_list)
    plt.ylabel('Log Cross Entropy')
    plt.title('Log Cross Entropy in function of the counting method')
    plt.show()


### For a given classifier, compute the perfomance metrics depending on min_df    
def fit_min_threshold(min_t, max_t, step, vectorizer, clf, raw_X, Y, k=5):
    thresholds = np.linspace(min_t, max_t, step)
    acc_list, ent_list = [], []
    best_acc, best_thresh = 0, min_t
    sparse_Y = scipy.sparse.csr_matrix(Y).transpose()
    for t in thresholds:
        vectorizer.set_params(min_df=t)
        X = get_vectors(vectorizer, raw_X)
        dataset = scipy.sparse.hstack([X, sparse_Y], format="csr")
        dataset = sk.utils.shuffle(dataset)
        acc, ent = cross_validation(clf, dataset, k)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
        acc_list.append(acc)
        ent_list.append(ent)

    plt.plot(thresholds, acc_list, 'ro')
    plt.xlabel('Minimum threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy in function of the minimum threshold')
    plt.show()
    plt.plot(thresholds, ent_list, 'ro')
    plt.xlabel('Minimum threshold')
    plt.ylabel('Log Cross Entropy')
    plt.title('Log Cross Entropy in function of the minimum threshold')
    plt.show()
    
    return best_thresh

### For a given classifier, compute the perfomance metrics depending on max_df
def fit_max_threshold(min_t, max_t, step, vectorizer, clf, raw_X, Y, k=5):
    thresholds = np.linspace(min_t, max_t, step)
    acc_list, ent_list = [], []
    best_acc, best_thresh = 0, min_t
    sparse_Y = scipy.sparse.csr_matrix(Y).transpose()
    for t in thresholds:
        vectorizer.set_params(max_df=t)
        X = get_vectors(vectorizer, raw_X)
        dataset = scipy.sparse.hstack([X, sparse_Y], format="csr")
        dataset = sk.utils.shuffle(dataset)
        acc, ent = cross_validation(clf, dataset, k)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
        acc_list.append(acc)
        ent_list.append(ent)

    plt.plot(thresholds, acc_list, 'ro')
    plt.xlabel('Maximum threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy in function of the maximum threshold')
    plt.show()
    plt.plot(thresholds, ent_list, 'ro')
    plt.xlabel('Maximum threshold')
    plt.ylabel('Log Cross Entropy')
    plt.title('Log Cross Entropy in function of the maximum threshold')
    plt.show()

    return best_thresh


### For a given classifier, compute the perfomance metrics depending on the stopwords list 
def test_stopwords(stopwords_list, names_list, vectorizer, clf, raw_X, Y, k=5):
    acc_list, ent_list = [], []
    sparse_Y = scipy.sparse.csr_matrix(Y).transpose()
    for s in stopwords_list:
        vectorizer.set_params(stop_words=s)
        X = get_vectors(vectorizer, raw_X)
        dataset = scipy.sparse.hstack([X, sparse_Y], format="csr")
        dataset = sk.utils.shuffle(dataset)
        acc, ent = cross_validation(clf, dataset, k)
        print(acc)
        acc_list.append(acc)
        ent_list.append(ent)

    plt.bar(names_list, acc_list)
    plt.ylabel('Accuracy')
    plt.title('Accuracy in function of the stopwords used')
    plt.show()
    plt.bar(names_list, ent_list)
    plt.ylabel('Log Cross Entropy')
    plt.title('Log Cross Entropy in function of the stopwords used')
    plt.show()


#########################################################################################
### Analysing the effect of the hyperparameters on the performance of the classifiers ###
#########################################################################################

### For a given naive bayes classifier, compute the perfomance metrics depending on the smoothing factor alpha
def fit_smoothing_nb(clf, dataset, step, k=5):
    smooth_lst = np.linspace(1.0e-8, 1, step)    # The list of the values of the parameter that we will test. We vary the parameter from 1e-8 to 1 testing 'step' values (i.e. 'step' is the number of points)
    acc_list, ent_list = [], []     # Lists storing the results
    best_acc, best_alpha = 0, 1.0e-8
    for a in smooth_lst:
        clf.set_params(alpha=a)     # Set the parameter to the next value
        acc, ent = cross_validation(clf, dataset, k)      # Train the model, then evaluate its performance using cross validation
        if acc > best_acc:
            best_acc = acc
            best_alpha = a
        acc_list.append(acc)
        ent_list.append(ent)

    # Plot the results
    plt.plot(smooth_lst, acc_list, 'ro')
    plt.xlabel('Smoothing factor')
    plt.ylabel('Accuracy')
    plt.title('Naive Bayes (Accuracy depending on Smoothing Factor)')
    plt.show()
    plt.plot(smooth_lst, ent_list, 'ro')
    plt.xlabel('Smoothing factor')
    plt.ylabel('Log Cross Entropy')
    plt.title('Naive Bayes (Log Cross Entropy depending on Smoothing Factor)')
    plt.show()

    return best_alpha

# Can be used for logistic regression or support vector machine
def fit_epsilon(min_eps, max_eps, clf, dataset, step, k=5):
    eps_lst = np.linspace(min_eps, max_eps, step)    # The list of the values of the parameter that we will test.
    acc_list, ent_list = [], []     # Lists storing the results
    best_acc, best_eps = 0, min_eps
    for e in eps_lst:
        clf.set_params(tol=e)     # Set the parameter to the next value
        acc, ent = cross_validation(clf, dataset, k)      # Train the model, then evaluate its performance using cross validation
        if acc > best_acc:
            best_acc = acc
            best_eps = e
        acc_list.append(acc)
        ent_list.append(ent)

    # Plot the results
    plt.plot(eps_lst, acc_list, 'ro')
    plt.xlabel('Tolerance for stopping criteria')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression (Accuracy depending on the Tolerance for stopping criteria)')
    plt.show()
    plt.plot(eps_lst, ent_list, 'ro')
    plt.xlabel('Tolerance for stopping criteria')
    plt.ylabel('Log Cross Entropy')
    plt.title('Logistic Regression (Log Cross Entropy depending on the Tolerance for stopping criteria)')
    plt.show()

    return best_eps

# Can be used for logistic regression or support vector machine
def fit_iter(min_iter, max_iter, clf, dataset, step, k=5):
    iter_lst = np.linspace(min_iter, max_iter, step)    # The list of the values of the parameter that we will test.
    acc_list, ent_list = [], []     # Lists storing the results
    best_acc, best_iter = 0, min_iter
    for i in iter_lst:
        clf.set_params(max_iter=i)     # Set the parameter to the next value
        acc, ent = cross_validation(clf, dataset, k)      # Train the model, then evaluate its performance using cross validation
        if acc > best_acc:
            best_acc = acc
            best_iter = i
        acc_list.append(acc)
        ent_list.append(ent)

    # Plot the results
    plt.plot(iter_lst, acc_list, 'ro')
    plt.xlabel('Maximum number of iteration')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression (Accuracy depending on the Maximum number of iteration)')
    plt.show()
    plt.plot(iter_lst, ent_list, 'ro')
    plt.xlabel('Maximum number of iteration')
    plt.ylabel('Log Cross Entropy')
    plt.title('Logistic Regression (Log Cross Entropy depending on the Maximum number of iteration)')
    plt.show()

    return best_iter


# Can be used for logistic regression or support vector machine
def fit_regu(min_c, max_c, clf, dataset, step, k=5):
    c_lst = np.linspace(min_c, max_c, step)    # The list of the values of the parameter that we will test.
    acc_list, ent_list = [], []     # Lists storing the results
    best_acc, best_c = 0, min_c
    for c in c_lst:
        clf.set_params(C=c)     # Set the parameter to the next value
        acc, ent = cross_validation(clf, dataset, k)      # Train the model, then evaluate its performance using cross validation
        if acc > best_acc:
            best_acc = acc
            best_c = c
        acc_list.append(acc)
        ent_list.append(ent)

    # Plot the results
    plt.plot(c_lst, acc_list, 'ro')
    plt.xlabel('Inverse of regularisation strength')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression (Accuracy depending on the Inverse of regularisation strength)')
    plt.show()
    plt.plot(c_lst, ent_list, 'ro')
    plt.xlabel('Inverse of regularisation strength')
    plt.ylabel('Log Cross Entropy')
    plt.title('Logistic Regression (Log Cross Entropy depending on the Inverse of regularisation strength)')
    plt.show()

    return best_c


def fit_number_adaboost(min_n, max_n, clf, dataset, step, k=5):
    n_lst = np.linspace(min_n, max_n, step)  # The list of the values of the parameter that we will test.
    acc_list, ent_list = [], []  # Lists storing the results
    best_acc, best_n = 0, min_n
    for n in n_lst:
        print(n)
        clf.set_params(n_estimators=math.floor(n))  # Set the parameter to the next value
        acc, ent = cross_validation(clf, dataset, k)  # Train the model, then evaluate its performance using cross validation
        if acc > best_acc:
            best_acc = acc
            best_n = math.floor(n)
        acc_list.append(acc)
        ent_list.append(ent)

    # Plot the results
    plt.plot(n_lst, acc_list, 'ro')
    plt.xlabel('Number of estimators')
    plt.ylabel('Accuracy')
    plt.title('Adaboost (Accuracy depending on Number of estimators)')
    plt.show()
    plt.plot(n_lst, ent_list, 'ro')
    plt.xlabel('Number of estimators')
    plt.ylabel('Log Cross Entropy')
    plt.title('Adaboost (Log Cross Entropy depending on Number of estimators)')
    plt.show()

    return best_n

def fit_learning_adaboost(min_lr, max_lr, clf, dataset, step, k=5):
    lr_lst = np.linspace(min_lr, max_lr, step)  # The list of the values of the parameter that we will test.
    acc_list, ent_list = [], []  # Lists storing the results
    best_acc, best_lr = 0, min_lr
    for l in lr_lst:
        clf.set_params(learning_rate=l)  # Set the parameter to the next value
        acc, ent = cross_validation(clf, dataset, k)  # Train the model, then evaluate its performance using cross validation
        if acc > best_acc:
            best_acc = acc
            best_lr = l
        acc_list.append(acc)
        ent_list.append(ent)

    # Plot the results
    plt.plot(lr_lst, acc_list, 'ro')
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.title('Adaboost (Accuracy depending on Learning rate)')
    plt.show()
    plt.plot(lr_lst, ent_list, 'ro')
    plt.xlabel('Learning rate')
    plt.ylabel('Log Cross Entropy')
    plt.title('Adaboost (Log Cross Entropy depending on Learning rate)')
    plt.show()

    return best_lr


def fit_number_random_forest(min_n, max_n, clf, dataset, step, k=5):
    n_lst = np.linspace(min_n, max_n, step)  # The list of the values of the parameter that we will test.
    acc_list, ent_list = [], []  # Lists storing the results
    best_acc, best_n = 0, min_n
    for n in n_lst:
        print(n)
        clf.set_params(n_estimators=math.floor(n))  # Set the parameter to the next value
        acc, ent = cross_validation(clf, dataset, k)  # Train the model, then evaluate its performance using cross validation
        if acc > best_acc:
            best_acc = acc
            best_n = math.floor(n)
        acc_list.append(acc)
        ent_list.append(ent)

    # Plot the results
    plt.plot(n_lst, acc_list, 'ro')
    plt.xlabel('Number of estimators')
    plt.ylabel('Accuracy')
    plt.title('Random Forest (Accuracy depending on Number of estimators)')
    plt.show()
    plt.plot(n_lst, ent_list, 'ro')
    plt.xlabel('Number of estimators')
    plt.ylabel('Log Cross Entropy')
    plt.title('Random Forest (Log Cross Entropy depending on Number of estimators)')
    plt.show()

    return best_n




def compare_penalty_svm(clf, dataset, penalties=['l1', 'l2'], k=5):
    clf.set_params(dual=False)
    acc_list, ent_list = [], []     # Lists storing the results
    for p in penalties:
        clf.set_params(penalty=p)     # Set the parameter to the next value
        acc, ent = cross_validation(clf, dataset, k)      # Train the model, then evaluate its performance using cross validation
        acc_list.append(acc)
        ent_list.append(ent)

    # Plot the results
    plt.bar(penalties, acc_list)
    plt.ylabel('Accuracy')
    plt.title('Support Vector Machine (Accuracy in function of the penalty)')
    plt.show()
    plt.bar(penalties, ent_list)
    plt.ylabel('Log Cross Entropy')
    plt.title('Support Vector Machine (Log Cross Entropy in function of the penalty)')
    plt.show()

def compare_loss_svm(clf, dataset, losses=['hinge', 'squared_hinge'], k=5):
    clf.set_params(dual=True)
    acc_list, ent_list = [], []     # Lists storing the results
    for l in losses:
        clf.set_params(loss=l)     # Set the parameter to the next value
        acc, ent = cross_validation(clf, dataset, k)      # Train the model, then evaluate its performance using cross validation
        acc_list.append(acc)
        ent_list.append(ent)

    # Plot the results
    plt.bar(losses, acc_list)
    plt.ylabel('Accuracy')
    plt.title('Support Vector Machine (Accuracy in function of the loss)')
    plt.show()
    plt.bar(losses, ent_list)
    plt.ylabel('Log Cross Entropy')
    plt.title('Support Vector Machine (Log Cross Entropy in function of the loss)')
    plt.show()

# Can be used for decision tree and random forest
def compare_criterion_tree(clf, dataset, criterions=['gini', 'entropy'], k=5):
    acc_list, ent_list = [], []  # Lists storing the results
    for c in criterions:
        clf.set_params(criterion=c)  # Set the parameter to the next value
        acc, ent = cross_validation(clf, dataset, k)  # Train the model, then evaluate its performance using cross validation
        acc_list.append(acc)
        ent_list.append(ent)

    # Plot the results
    plt.bar(criterions, acc_list)
    plt.ylabel('Accuracy')
    plt.title('Random Forest (Accuracy in function of the criterion)')
    plt.show()
    plt.bar(criterions, ent_list)
    plt.ylabel('Log Cross Entropy')
    plt.title('Random Forest (Log Cross Entropy in function of the criterion)')
    plt.show()

def compare_splitter_tree(clf, dataset, splits=['best', 'random'], k=5):
    acc_list, ent_list = [], []  # Lists storing the results
    for s in splits:
        clf.set_params(splitter=s)  # Set the parameter to the next value
        acc, ent = cross_validation(clf, dataset, k)  # Train the model, then evaluate its performance using cross validation
        acc_list.append(acc)
        ent_list.append(ent)

    # Plot the results
    plt.bar(splits, acc_list)
    plt.ylabel('Accuracy')
    plt.title('Decision Tree (Accuracy in function of the splitter)')
    plt.show()
    plt.bar(splits, ent_list)
    plt.ylabel('Log Cross Entropy')
    plt.title('Decision Tree (Log Cross Entropy in function of the splitter)')
    plt.show()

####################################################################################
### Analyze the performance of the different classifiers and choose the best one ###
####################################################################################

### Test different classifiers and compare their results
def compare_classifiers(clf_list, clf_names, dataset, k=5):
    acc_list, ent_list = [], []
    i = 0
    for c in clf_list:
        print(clf_names[i])
        i += 1
        acc, ent = cross_validation(c, dataset, k)
        acc_list.append(acc)
        ent_list.append(ent)
    
    plt.bar(clf_names, acc_list)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of the different classifiers')
    plt.show()
    plt.bar(clf_names, ent_list)
    plt.ylabel('Log Cross Entropy')
    plt.title('Log Cross Entropy of the different classifiers')
    plt.show()

    return acc_list, ent_list

### Final Training / Testing
def evaluate_model(clf, train_set, test_set):
    res = test_classifier(clf, train_set, test_set)
    mat = confusion_matrix(res, test_set[:, -1])
    print("Confusion Matrix : ", mat)
    return evaluate(mat, True)

def final_evaluation(clf_list, train_set, test_set):
    acc_list, ent_list = [], []
    i = 0
    for c in clf_list:
        print(clf_names[i])
        i += 1
        acc, ent = evaluate_model(c, train_set, test_set)
        acc_list.append(acc)
        ent_list.append(ent)

    plt.bar(clf_names, acc_list)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of the different classifiers')
    plt.show()
    plt.bar(clf_names, ent_list)
    plt.ylabel('Log Cross Entropy')
    plt.title('Log Cross Entropy of the different classifiers')
    plt.show()

    return acc_list, ent_list




###################################
### Application to the datasets ###
###################################

## List of the classifiers

mnb = nb.MultinomialNB()
cnb = nb.ComplementNB(alpha=0.315)
lr = sk.linear_model.LogisticRegression(solver='lbfgs', max_iter=300, multi_class='multinomial')
svm = sk.svm.LinearSVC()
dtr = sktree.DecisionTreeClassifier()
rfc = skens.RandomForestClassifier()
ada = skens.AdaBoostClassifier()

clf_list = [mnb, cnb, lr, svm, dtr, rfc, ada]
clf_names = ['Multinomial Naive Bayes', 'Complement Naive Bayes', 'Logistic Regression', 'Support Vector Machine',
             'Decision Tree', 'Random Forest', 'AdaBoost']

## Preprocessing parameters

max_thresh = 0.9
min_thresh = 1e-4

def naive_tokenizer(s):
    return s.split()

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def symbol_tokenize(s): return re_tok.sub(r' \1 ', s).split()

token_list = [None, naive_tokenizer, symbol_tokenize]
token_names = ['Default', 'Naive', 'Symbol removal']

sw_lst = [None, 'english', stopwords.words('english')]
sw_names = ['None', 'Scikit.learn', 'NLTK']

vectorizer = sk.feature_extraction.text.CountVectorizer(stop_words='english', max_df=max_thresh, min_df=min_thresh)


### 20 NEWS GROUP DATASET ###
"""
twenty_train = fetch_20newsgroups(subset='train', remove=(['headers', 'footers', 'quotes']))

classes, occ = np.unique(twenty_train.target, return_counts=True)

plt.barh(twenty_train.target_names, occ)
plt.ylabel('Number of occurences')
plt.title('Distribution of the different classes')
plt.show()

### Choose the preprocessing parameters

best_ngram(1, 2, vectorizer, cnb, twenty_train.data, twenty_train.target)
test_tokenizer(token_list, token_names, vectorizer, cnb, twenty_train.data, twenty_train.target)
test_counting(vectorizer, cnb, twenty_train.data, twenty_train.target)
print(fit_min_threshold(0, 0.001, 10, vectorizer, cnb, twenty_train.data, twenty_train.target, 5))
print(fit_max_threshold(0.5, 1, 20, vectorizer, cnb, twenty_train.data, twenty_train.target, 5))
test_stopwords(sw_lst, sw_names, vectorizer, cnb, twenty_train.data, twenty_train.target, 5)

### Find the hyperparameters of the classifier


X = get_vectors(vectorizer, twenty_train.data)
Y = scipy.sparse.csr_matrix(twenty_train.target).transpose()
dataset = scipy.sparse.hstack([X, Y], format="csr")
dataset = sk.utils.shuffle(dataset)

print(fit_smoothing_nb(mnb, dataset, 20))

print(fit_smoothing_nb(cnb, dataset, 20))

print(fit_epsilon(1.e-4, 1.e-3, lr, dataset, 5))
fit_iter(300, 700, lr, dataset, 3)
print(fit_regu(0.1, 10, lr, dataset, 10))

print(fit_epsilon(1.e-4, 2.e-3, svm, dataset, 20))
fit_iter(800, 1500, svm, dataset, 8)
print(fit_regu(0.1, 2, svm, dataset, 10))
compare_penalty_svm(svm, dataset)
compare_loss_svm(svm, dataset)

compare_criterion_tree(dtr, dataset)
compare_splitter_tree(dtr, dataset)

compare_criterion_tree(rfc, dataset)
print(fit_number_random_forest(10, 50, ada, dataset, 5))

print(fit_number_adaboost(10, 50, ada, dataset, 5))
print(fit_learning_adaboost(0.75, 1.25, ada, dataset, 5))


print(compare_classifiers(clf_list, clf_names, dataset))


### Final test

mnb = nb.MultinomialNB(alpha=0.05)
cnb = nb.ComplementNB(alpha=0.315)
lr = sk.linear_model.LogisticRegression(solver='lbfgs', tol=1e-3, C=6.7, max_iter=300, multi_class='multinomial')
svm = sk.svm.LinearSVC(tol=1e-3, C=0.3)
dtr = sktree.DecisionTreeClassifier()
rfc = skens.RandomForestClassifier(n_estimators=30)
ada = skens.AdaBoostClassifier(n_estimators=30, learning_rate=1.25)

clf_list = [mnb, cnb, lr, svm, dtr, rfc, ada]
clf_names = ['Multinomial Naive Bayes', 'Complement Naive Bayes', 'Logistic Regression', 'Support Vector Machine',
             'Decision Tree', 'Random Forest', 'AdaBoost']

twenty_test = fetch_20newsgroups(subset='test', remove=(['headers', 'footers', 'quotes']))

X_test = get_vectors(vectorizer, twenty_test.data, fit=False)
Y_test = scipy.sparse.csr_matrix(twenty_test.target).transpose()
test_set = scipy.sparse.hstack([X_test, Y_test], format="csr")

print(final_evaluation(clf_list, dataset, test_set))

### Optimize the best classifier

text_mnb = Pipeline([('vect', vectorizer), ('tfidf', TfidfTransformer()), ('clf', mnb)])

params = {'vect__ngram_range': [(1, 1), (2, 2)], 'vect__stop_words': [None, 'english', stopwords.words('english')], 'vect__min_df': [0, 5e-5, 1e-4] ,'tfidf__use_idf': (True, False), 'clf__alpha': [1, 0.8, 0.6, 0.4, 0.3, 0.2], 'clf__fit_prior': (True, False)}

gs_mnb = GridSearchCV(text_mnb, params, cv=5, n_jobs=-1)
gs_mnb = gs_mnb.fit(twenty_train.data, twenty_train.target)

for param_name in sorted(params.keys()):
    print("%s: %r" % (param_name, gs_mnb.best_params_[param_name]))

res = gs_mnb.predict(twenty_test.data)
mat = confusion_matrix(res, twenty_test.target, False)
evaluate(mat, True)
"""
### IMDB DATASET ###

ps = PorterStemmer()

def get_text(directory, stemmer):
    X = []
    for filename in os.listdir(directory):
        file = open(directory + '\\' + filename, "r", encoding="utf8")
        for l in file:
            X.append(stemmer.stem(l))
    return X

train_neg = get_text(".\\train\\neg", ps)
train_pos = get_text(".\\train\\pos", ps)
test_neg = get_text(".\\test\\neg", ps)
test_pos = get_text(".\\test\\pos", ps)

train = train_neg + train_pos
test = test_neg + test_pos

train_y = [0 for i in range(len(train_neg))] + [1 for i in range(len(train_pos))]
test_y = [0 for i in range(len(test_neg))] + [1 for i in range(len(test_pos))]


max_thresh = 0.99
min_thresh = 0

vectorizer = sk.feature_extraction.text.CountVectorizer(min_df=min_thresh, max_df=max_thresh)

### Choose the preprocessing parameters

#best_ngram(1, 2, vectorizer, cnb, train, train_y)
#vectorizer.set_params(ngram_range=(1, 1))
test_tokenizer(token_list, token_names, vectorizer, cnb, train, train_y)
vectorizer.set_params(tokenizer=None)
#test_counting(vectorizer, cnb, train, train_y)
print(fit_min_threshold(0, 0.15, 10, vectorizer, cnb, train, train_y, 5))
vectorizer.set_params(min_df=min_thresh)
print(fit_max_threshold(0.5, 0.75, 20, vectorizer, cnb, train, train_y, 5))
vectorizer.set_params(min_df=max_thresh)
test_stopwords(sw_lst, sw_names, vectorizer, cnb, train, train_y.target, 5)

X = get_vectors(vectorizer, train)
Y = scipy.sparse.csr_matrix(train_y).transpose()
dataset = scipy.sparse.hstack([X, Y], format="csr")
dataset = sk.utils.shuffle(dataset)


### Find the hyperparameters of the classifier

#print(fit_smoothing_nb(mnb, dataset, 20))

#print(fit_smoothing_nb(cnb, dataset, 20))

print(fit_epsilon(1.e-4, 1.e-3, lr, dataset, 5))
fit_iter(300, 700, lr, dataset, 5)
#print(fit_regu(0.1, 10, lr, dataset, 10))

print(fit_epsilon(1.e-4, 2.e-3, svm, dataset, 20))
fit_iter(800, 1500, svm, dataset, 8)
#print(fit_regu(0.1, 2, svm, dataset, 10))
#compare_penalty_svm(svm, dataset)
#compare_loss_svm(svm, dataset)

#compare_criterion_tree(dtr, dataset)
#compare_splitter_tree(dtr, dataset)

#compare_criterion_tree(rfc, dataset)
print(fit_number_random_forest(10, 50, ada, dataset, 5))

print(fit_number_adaboost(10, 50, ada, dataset, 3))
print(fit_learning_adaboost(0.75, 1.25, ada, dataset, 5))


print(compare_classifiers(clf_list, clf_names, dataset))

### Final test

mnb = nb.MultinomialNB(alpha=0.25)
cnb = nb.ComplementNB(alpha=0.25)
lr = sk.linear_model.LogisticRegression(solver='lbfgs', tol=1e-3, C=6.7, max_iter=300, multi_class='multinomial')
svm = sk.svm.LinearSVC(tol=1e-3, C=0.3)
dtr = sktree.DecisionTreeClassifier()
rfc = skens.RandomForestClassifier(n_estimators=30)
ada = skens.AdaBoostClassifier(n_estimators=30, learning_rate=1.25)

X_test = get_vectors(vectorizer, test, fit=False)
Y_test = scipy.sparse.csr_matrix(test_y).transpose()
test_set = scipy.sparse.hstack([X_test, Y_test], format="csr")

#print(final_evaluation(clf_list, dataset, test_set))

### Optimize the best classifier


text_lr = Pipeline([('vect', vectorizer), ('tfidf', TfidfTransformer()), ('logistic', lr)])

print(text_lr.get_params().keys())

params = {'vect__ngram_range': [(1, 1), (2, 2)], 'vect__stop_words': ['english', stopwords.words('english')], 'vect__min_df': [0, 1e-4] ,'tfidf__use_idf': (True, False), 'logistic__tol': [1e-3, 1e-4], 'logistic__max_iter': [300, 500], 'logistic__C': [0.1, 1, 5, 10]}

gs_lr = GridSearchCV(text_lr, params, cv=5, n_jobs=-1)
gs_lr = gs_lr.fit(train, train_y)

for param_name in sorted(params.keys()):
    print("%s: %r" % (param_name, gs_lr.best_params_[param_name]))

res = gs_lr.predict(test)
mat = confusion_matrix(res, test_y, False)
evaluate(mat, True)





