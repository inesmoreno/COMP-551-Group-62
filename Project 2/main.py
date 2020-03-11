import numpy as np
import nltk
import scipy
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
import sklearn as sk
import sklearn.naive_bayes as nb
import math
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups


#nltk.download()

"""
To do list :
- Get the actual databases that we want to work on, and get the results on these databases  --> First one Ok
- Adjust a few things in the preprocessing part (use frequencies and sparse matrices)       --> Ok
- Change the evaluations metrics to fit classification (multiple classes)                   --> Ok
- /!\ Add the missing classifiers to the list of classifiers that we want to test
- /!\ Add tests for other hyperparameters
- Add something to select the hyperparameters                                               --> Ok
- Add some sort of validation test (cross-validation most likely)                           --> Ok
- Adjust the tests for the best preprocessing parameters                                    --> Must check the format of the other dataset + stemmer
- Create a global test that automatically fit the parameters                                --> Maybe, kinda bothersome since we have to do one for each classifiers and the running time is quite long
"""

############################################################################
### Bonus part : preprocessing of the text to get more relevant features ###
############################################################################
""" Not adapted right not, might delete later """
### Text processing : apply stemmers/lemmatizers on the text and separate each line

def preprocessing_stem(filename, stemmer):
    file = open(filename, "r")
    X = []
    for l in file:
        stemmed = ''
        for w in l:
            stemmed += stemmer.stem(w)
        X.append(stemmed)
    return X

##########################################################
### Transforming the text into vectors + preprocessing ###
##########################################################

### Feature extraction : transform each line into a vector of features
def get_vectors(vectorizer, raw_X, frequency=True, downscale=False, fit=True):
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
def confusion_matrix(predicted, labels):
    correct = np.ravel(labels.toarray())
    classes = np.unique(correct)        # List the different labels found in the testing set
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

### For a given classifier, compute the perfomance metrics depending on min_df    
""" This algorithm plot 'step' points of the average performance of a classifier 'clf' using k-fold cross validation when we vary the parameter 'min_df' from 'min_t' to 'max_t' """
def fit_min_threshold(min_t, max_t, step, vectorizer, clf, raw_X, Y, k=5):
    thresholds = np.linspace(min_t, max_t, step)
    acc_list, ent_list = [], []
    best_acc, best_thresh = 0, min_t
    sparse_Y = scipy.sparse.csr_matrix(Y).transpose()
    for t in thresholds:
        vectorizer.set_params(min_df=t)
        X = get_vectors(vectorizer, raw_X)
        dataset = scipy.sparse.hstack([X, sparse_Y], format="csr")
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


### For a given classifier, compute the perfomance metrics depending on the stopwords list 
def test_stopwords(stopwords_list, names_list, vectorizer, clf, raw_X, Y, k=5):
    acc_list, ent_list = [], []
    sparse_Y = scipy.sparse.csr_matrix(Y).transpose()
    for s in stopwords_list:
        vectorizer.set_params(stop_words=s)
        X = get_vectors(vectorizer, raw_X)
        dataset = scipy.sparse.hstack([X, sparse_Y], format="csr")
        acc, ent = cross_validation(clf, dataset, k)
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

""" /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ """
""" This is the part we want to copy paste to create new test. Tell me if something is unclear there    """
""" We most likely also want to add something so that the algorithm return the best hyperparameter      """
""" /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ """

### For a given naive bayes classifier, compute the perfomance metrics depending on the smoothing factor alpha 
""" For numerical parameters """
def fit_smoothing_nb(clf, dataset, step, k=5):
    smooth_lst = np.linspace(1.0e-8, 1, step)    # The list of the values of the parameter that we will test. We vary the parameter from 0 to 1 testing 'step' values (i.e. 'step' is the number of points)
    acc_list, ent_list = [], []     # Lists storing the results
    best_acc, best_alpha = 0, 0
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


""" For discrete parameters """
def compare_penalty_svm(clf, dataset, penalties=['l1', 'l2'], k=5):
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



####################################################################################
### Analyze the performance of the different classifiers and choose the best one ###
####################################################################################

### Test different classifiers and compare their results
def compare_classifiers(clf_list, clf_names, dataset, k=5):
    acc_list, ent_list = [], []
    for c in clf_list:
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

### Final Training / Testing
def evaluate_model(clf, train_set, test_set):
    res = test_classifier(clf, train_set, test_set)
    mat = confusion_matrix(res, test_set[:, -1])
    print("Confusion Matrix : ", mat)
    evaluate(mat, True)



###################################
### Application to the datasets ###
###################################


twenty_train = fetch_20newsgroups(subset='train', remove=(['headers', 'footers', 'quotes']))
vectorizer_classic = sk.feature_extraction.text.CountVectorizer()

sw_lst = [None, 'english', stopwords.words('english')]
sw_names = ['None', 'Scikit.learn', 'NLTK']

mnb = nb.MultinomialNB(alpha=0.8)
cnb = nb.ComplementNB(alpha=0.84)
lr = sk.linear_model.LogisticRegression(solver='lbfgs', max_iter=500, multi_class='multinomial')
svm = sk.svm.LinearSVC(dual=False)

clf_list = [mnb, cnb, lr, svm]
clf_names = ['Multinomial Naive Bayes', 'Complement Naive Bayes', 'Logistic Regression', 'Support Vector Machine']

### Choose the preprocessing parameters

#fit_min_threshold(0, 0.25, 5, vectorizer_classic, cnb, twenty_train.data[:600], twenty_train.target[:600], 5)

### Find the hyperparameters of the classifier

X = get_vectors(vectorizer_classic, twenty_train.data)

Y = scipy.sparse.csr_matrix(twenty_train.target).transpose()
dataset = scipy.sparse.hstack([X, Y], format="csr")
dataset = sk.utils.shuffle(dataset)

#compare_penalty_svm(svm, dataset)

compare_classifiers(clf_list, clf_names, dataset)


### Final test
"""
twenty_test = fetch_20newsgroups(subset='test', remove=(['headers', 'footers', 'quotes']))
X_test = get_vectors(vectorizer_classic, twenty_test.data, fit=False)
Y_test = scipy.sparse.csr_matrix(twenty_test.target).transpose()
test_set = scipy.sparse.hstack([X_test, Y_test], format="csr")
print(dataset.shape)
print(test_set.shape)

evaluate_model(cnb, dataset, test_set)
"""


""" This concerns the dataset from my previous assignment, not relevant """


"""
#######################
# Definition of the classifiers and the tools used for preprocessing
def naive_tokenizer(s):
    return s.split()

ps = PorterStemmer()
#######################


neg_X = preprocessing_stem('.\\rt-polaritydata\\rt-polaritydata\\rt-polarity.neg', ps)
pos_X = preprocessing_stem('.\\rt-polaritydata\\rt-polaritydata\\rt-polarity.pos', ps)
raw_X = neg_X + pos_X

pst = PunktSentenceTokenizer(raw_X)

######################

svm1 = sk.svm.LinearSVC(penalty='l2', loss='hinge', max_iter=2000)
svm2 = sk.svm.LinearSVC(loss='squared_hinge', max_iter=2000)
svm3 = sk.svm.LinearSVC(dual=False, max_iter=2000)
"""