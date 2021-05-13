# Justin Gallagher and Craig Mcghee
# Titanic Decision Tree
# Project #3
# Highest Accuracy: 92.22222222222223
# Highest Kaggle Score: 0.78229

import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'

# grab data from the csv files
train_ = pd.read_csv("Titanic_training.csv")
test_ = pd.read_csv("Titanic_test.csv")

# this function processes the training data set so that it is easier to work with
def process_train(train):
    # replace 'female' with '0' and 'male' with '1'
    train['Sex'] = train['Sex'].replace('female', 0)
    train['Sex'] = train['Sex'].replace('male', 1)

    # add SizeOfFamily feature, which equals SibSp (siblings/spouse) + Parch (parents/children)
    train['SizeOfFamily'] = train['SibSp'] + train['Parch']

    # add NonMaster feature, where NonMaster = 1 if 'Master' appears in name and NonMaster = 0 otherwise
    train['NonMaster'] = 0
    for i in range(len(train.Name)):
        if "Master" in train.Name[i]:
            train.NonMaster[i] = 1

    # create SomeFamily and NoFamily to represent whether a person has living family or not
    train['SomeFamily'] = 0
    train['NoFamily'] = 0

    # take the surname and trim it so that it is just the last name
    train['Surname'] = train.Name.str.extract("([A-Z]\w{0,})")

    # figure out if there is family that is still alive and set the SomeFamily flag
    for i in range(len(train.Surname)):
        for j in range(i + 1, len(train.Surname)):
            if train.Surname[i] == train.Surname[j] and (train.Survived[i] == 1 or train.Survived[j]):
                train.SomeFamily[i] = 1
                train.SomeFamily[j] = 1

    # determine if no family exists, and set the appropriate NoFamily flag
    for i in range(len(train.Surname)):
        for j in range(i + 1, len(train.Surname)):
            if train.Surname[i] == train.Surname[j] and train.SomeFamily[i] == 0:
                train.NoFamily[i] = 1
                train.NoFamily[j] = 1

    # from the set, drop the labels that do not matter
    train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked'], axis=1)
    print('Processed Training Set:')
    print(train)
    return train

# this function processes the test set of data to make it easier to work with
def process_test(test):
    # determine average age and fare
    average_age = int(test.Age.median())
    fare = float(test_.Fare.median())

    # Fields that are empty are filled in
    test['Embarked'] = test['Embarked'].fillna("S")
    test['Age'] = test['Age'].fillna(average_age)
    test['Fare'] = test['Fare'].fillna(fare)

    # replace 'female' with '0' and 'male' with '1'
    test['Sex'] = test['Sex'].replace('female', 0)
    test['Sex'] = test['Sex'].replace('male', 1)

    # add SizeOfFamily feature, which equals SibSp (siblings/spouse) + Parch (parents/children)
    test['SizeOfFamily'] = test['SibSp'] + test['Parch']

    # add NonMaster feature, where NonMaster = 1 if 'Master' appears in name and NonMaster = 0 otherwise
    test['NonMaster'] = 0
    for i in range(len(test.Name)):
        if "Master" in test.Name[i]:
            test.NonMaster[i] = 1

    # take the surname and trim it so that it is just the last name
    test['Surname'] = test.Name.str.extract("([A-Z]\w{0,})")

    # from the set, drop the labels that do not matter
    test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked'], axis=1)
    print('Processed Test Set:')
    print(test)
    return test

# this function takes the training and test set, calculates new SomeFamily and NoFamily values,
# and then returns a new test set
def combine(train, test):
    test['SomeFamily'] = 0
    test['NoFamily'] = 0
    for i in range(len(test.Surname)):
        for j in range(len(train.Surname)):
            if test.Surname[i] == train.Surname[j]:
                if train.SomeFamily[j] == 1:
                    test.SomeFamily[i] = 1

                if train.NoFamily[j] == 1:
                    test.NoFamily[i] = 1
    print(test)
    return test

# main function
def main ():
    
    # grab the passenger id for the submission output
    pass_id = pd.DataFrame(test_['PassengerId'])

    # process the data from the training and test sets
    print("Processing training data set...")
    train_process = process_train(train_)
    print("\nProcessing test data set...")
    test_process = process_test(test_)
    print("\nCombining data sets...")
    test_process = combine(train_process, test_process)

    # drop non-numerical values from the train and test sets
    train_process = train_process.drop(['Surname'], axis=1)
    test_process = test_process.drop(['Surname'], axis=1)

    # create a split list,
    train, test = np.split(train_process.sample(frac=1), [int(0.8 * len(train_process))])

    # drop non-numerical values from the training set
    y_train = train['Survived']
    x_train = train.drop(['Survived'], axis=1)

    # drop non-numerical values from the training set
    y_test = test['Survived']
    x_test = test.drop(['Survived'], axis=1)

    # create a Decision Tree and fit it with the training set
    tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    tree_model.fit(x_train, y_train)
    print("\n")
    print("Text-Based Decision Tree:")
    print(tree.export_text(tree_model))

    # take the predicted result, and join it with the passenger id to create the submission file
    predicted = pd.DataFrame({'Survived': tree_model.predict(test_process)})
    result = pass_id.join(predicted)
    result.to_csv("decision_tree_submission.csv", index=False)

    # determine accuracy of the tree
    train_acc = tree_model.score(x_train, y_train) * 100
    test_acc = tree_model.score(x_test, y_test) * 100
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)

    # create a .dot file to be converted to .png
    tree.export_graphviz(tree_model,
                         out_file="decision_tree.dot",
                         feature_names=list(train.drop(['Survived'], axis=1)),
                         class_names=['Died', 'Survived'],
                         filled=True)

    # determine number of nodes within the tree
    number_nodes = tree_model.tree_.node_count
    print("Number of nodes in tree = ", number_nodes)

    # determine number of leaf nodes within tree
    number_leaf_nodes = tree_model.tree_.n_leaves
    print("Number of leaf nodes in tree = ", number_leaf_nodes)

main()