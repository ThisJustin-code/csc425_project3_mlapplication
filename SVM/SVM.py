# Justin Gallagher and Craig Mcghee
# SVM MNIST
# Project #3

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'

# this function will show random samples from the test set to help visualize the data
# (the line the calls this function is commented out by default)
def show_samples(X_test, y_predicted, expected, num_of_samples):
    random_samples = []
    for i in range(num_of_samples):
        r = np.random.randint(0, X_test.shape[0])
        random_samples.append(r)
    title = "Predicted: {} / Expected: {}"
    for sample in random_samples:
        number = X_test.iloc[sample, :]
        plt.imshow(number.values.reshape(28, 28), cmap='gray')
        plt.title(title.format(y_predicted[sample], expected[sample]))
        plt.show()

# this function trains on a data set using svm classifier with varying train/test ratios and kernels
def svm_train_test(X, y, train_test_ratio, kernel):
    # split X and y into random train and test subsets with the given train_test_ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_ratio, random_state=30, stratify=y)

    X_train[X_train > 0] = 1
    X_test[X_test > 0] = 1

    # begin SVM classification training
    print('Begin Kernel SVM Classification - [Ratio:{} Kernel:{}]'.format(train_test_ratio, kernel))
    start = time.time()
    print('Start training... [{:.2f}]'.format(start))
    model = svm.SVC(C=10, gamma=0.001, kernel=kernel)
    model.fit(X_train, y_train)
    finish = time.time()
    print('Stop training... [{:.2f}]'.format(finish))
    total_time = finish - start
    print('Total training time: {:.2f}'.format(total_time))

    # begin testing
    print('Begin Testing...')
    expected = y_test
    y_predicted = model.predict(X_test)
    text = "Classification report for Classifier {}:\n{}"
    print(text.format(model, metrics.classification_report(expected, y_predicted)))
    print("Precise Kernel SVM Classification - [Ratio:{} Kernel:{}] Accuracy = {}\n\n\n\n\n".format(train_test_ratio, kernel, metrics.accuracy_score(expected, y_predicted)))
    # the following line will call a function that shows samples from the test set for visual aid (is not necessary)
    #show_samples(X_test, y_predicted, expected, 5)
    return [kernel, "{:.0f}".format(float(train_test_ratio * 100)), "{:.2f}".format(float(metrics.accuracy_score(expected, y_predicted) * 100))]

def main():
    # grab mnist data from the csv file
    mnist_data = pd.read_csv('mnist_subset.csv', sep=',', index_col=0)

    # grab data from mnist_data and store in X, grab row labels from mnist_data, store in y
    X = mnist_data.iloc[:,:]
    y = mnist_data.index

    # lists to hold metrics data from training and tests, used for plotting data
    metrics_list = []
    acc_linear_list = []
    ratio_linear_list = []
    acc_poly_list = []
    ratio_poly_list = []
    acc_rbf_list = []
    ratio_rbf_list = []

    # train/test ratio should be in form 0 < i < 1, where i is the percentage of data to be tested,
    # and the remaining data will be used for training.
    # kernals used in this program will include 'linear', 'poly', and 'rbf'
    metrics_list.append(svm_train_test(X, y, 0.05, 'linear'))
    metrics_list.append(svm_train_test(X, y, 0.1, 'linear'))
    metrics_list.append(svm_train_test(X, y, 0.2, 'linear'))
    metrics_list.append(svm_train_test(X, y, 0.3, 'linear'))
    metrics_list.append(svm_train_test(X, y, 0.5, 'linear'))
    metrics_list.append(svm_train_test(X, y, 0.6, 'linear'))
    metrics_list.append(svm_train_test(X, y, 0.8, 'linear'))
    metrics_list.append(svm_train_test(X, y, 0.05, 'poly'))
    metrics_list.append(svm_train_test(X, y, 0.1, 'poly'))
    metrics_list.append(svm_train_test(X, y, 0.2, 'poly'))
    metrics_list.append(svm_train_test(X, y, 0.3, 'poly'))
    metrics_list.append(svm_train_test(X, y, 0.5, 'poly'))
    metrics_list.append(svm_train_test(X, y, 0.6, 'poly'))
    metrics_list.append(svm_train_test(X, y, 0.8, 'poly'))
    metrics_list.append(svm_train_test(X, y, 0.05, 'rbf'))
    metrics_list.append(svm_train_test(X, y, 0.1, 'rbf'))
    metrics_list.append(svm_train_test(X, y, 0.2, 'rbf'))
    metrics_list.append(svm_train_test(X, y, 0.3, 'rbf'))
    metrics_list.append(svm_train_test(X, y, 0.5, 'rbf'))
    metrics_list.append(svm_train_test(X, y, 0.6, 'rbf'))
    metrics_list.append(svm_train_test(X, y, 0.8, 'rbf'))

    # extract data from the metrics list and split into lists according to kernel
    metrics_list = np.array(metrics_list)
    for i in range(len(metrics_list)):
        if metrics_list[i, 0] == 'linear':
            acc_linear_list.append(float(metrics_list[i, 1]))
            ratio_linear_list.append(float(metrics_list[i, 2]))
        if metrics_list[i, 0] == 'poly':
            acc_poly_list.append(float(metrics_list[i, 1]))
            ratio_poly_list.append(float(metrics_list[i, 2]))
        if metrics_list[i, 0] == 'rbf':
            acc_rbf_list.append(float(metrics_list[i, 1]))
            ratio_rbf_list.append(float(metrics_list[i, 2]))

    # plot the accuracy results from the test set for the different kernels used
    fig, axs = plt.subplots(3)
    axs[0].set_title('Linear Accuracy vs. Ratio')
    axs[0].plot(acc_linear_list, ratio_linear_list, 'ro')
    axs[1].set_title('Poly Accuracy vs. Ratio')
    axs[1].plot(acc_poly_list, ratio_poly_list, 'bs')
    axs[2].set_title('RBF Accuracy vs. Ratio')
    axs[2].plot(acc_rbf_list, ratio_rbf_list, 'g^')
    plt.show()

    # plot the accuracy results from the entire test set
    plt.scatter(acc_linear_list, ratio_linear_list)
    plt.scatter(acc_poly_list, ratio_poly_list)
    plt.scatter(acc_rbf_list, ratio_rbf_list)
    plt.show()

main()