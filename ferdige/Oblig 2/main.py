import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import plot_tree
from xgboost import XGBClassifier

# Oversampling and under sampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter






# import dataset for diabetes
dataset = pd.read_csv("diabetes.csv")
# dataset = ((dataset - dataset.min() ) / (dataset.max() - dataset.min()))

# Check dataset for values, if non-null etc..
print(dataset.info())

# show firs 5 rows in dataset:
print(dataset.head())


# Target the columns for x and y.

# x = inputs
x_input_columns = dataset.drop(["Outcome"], axis=1)

print("x input: \n", x_input_columns.head())


# y = targeted column... diabetes yes/no = 1/0
y_target_column = dataset["Outcome"]


# Take care of imbalanced data:
# Randomly over sample the minority class

print("y target: \n", y_target_column.head())

print("Diabetes description: \n", dataset.describe())
#sns.pairplot(dataset, hue='Outcome', palette='Dark2')


# Split dataset in 70% train and 30% test
x_train, x_test, y_train, y_test = train_test_split(x_input_columns, y_target_column, test_size=0.2, random_state=110)

# fix imbalanced datasets with oversampling
# ros = RandomOverSampler(random_state=44)
# X_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)
# Result: Bad performance, worse accuracy than the original method, but still better than UnderSampling.

# fix imbalanced dataset with undersmapling
# rus = RandomUnderSampler(random_state=42)
# X_train_rus, y_train_rus = rus.fit_resample(x_train, y_train)
# Result: Bad performance, worse accuracy than ever, with a bad confusion matrix.


modeller = []

# ML models
# support Vector Machines
#svm_model = svm.SVC(kernel="linear") #sigmoid = 0.54 acc, poly=0.81 rbf = 0.79, linear= 0.84 m/ c=10, linear = 0.81 m/c=0.1 linear = 0.83 m/random_state..
svm_model = make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=44))
#modeller.append(svm_model)

# check if props are missing numbers
missing_props = dataset.isna().mean(axis=0)
print(missing_props) # output = oK verified, no missing values.

# Decision trees
tree_model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=10, splitter="best")
#modeller.append(tree_model)

# Naive Bayes Gaussian
naive_model = GaussianNB()
modeller.append(naive_model)

# XGBC calssifier
XGBC_model = XGBClassifier(eval_metric='mlogloss') # eval_metric='mlogloss'
#modeller.append(XGBC_model)


""" Check correlation between non diabetics and diabetics from parameters as Age, Bmi etc..
sns.jointplot(x='Age', y='BMI', data=dataset, kind="reg")
"""

""" check BMI vs AGE
g = sns.jointplot(data=dataset, x="Age", y="BMI")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
plt.show()


# check BMI vs AGE
g = sns.jointplot(data=dataset, x="BMI", y="BloodPressure", hue="Outcome")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
plt.show()
"""



# Test the accuracy of the model by looping through the array of models we created earlier
for model in modeller:
    print(f"Model: {model}")
    # Fit and train the model with the x train and y train set.
    model.fit(x_train, y_train)
    # predict the results with x_test set
    y_prediction = model.predict(x_test)

    # get the accuracy score
    accuracy = accuracy_score(y_test, y_prediction)
    # print results
    print(f"Accuracy of {model} is {accuracy}\n")
    # check for true positives, false negatives and true negatives & false positives.
    confusion = confusion_matrix(y_test, y_prediction)
    crossValidation = cross_val_score(model, x_train, y_train, cv=5)
    print(f"Confusion Matrix of {model} is: \n{confusion}")
    print(f"Cross Validation Performance results: \n"
          f"{crossValidation}\n"
          f"Cross Validation mean: {crossValidation.mean()} accuracy with a standard "
          f"deviation of {crossValidation.std()}")
    print('Train data accuracy: ', round(model.score(x_train, y_train), 5))
    print('Test data accuracy: ', round(model.score(x_test, y_test), 5))
    print("\n\n\n\n")


"""
    For previewing Decision Tree nodes..
    plt.figure(figsize=(20, 20))
    plot_tree(model)
    plt.show()
"""


"""
Kernel SVC = Linear
Kernel Decision Tree = Entropy

Results: 


model 1. 
    SVC model has an Accuracy output of: 0.7835497835497836
    
    Confusion Matrix of SVC model is: 
                                        [[134  19]
                                        [ 31  47]]
        True positives: 134
        False positives: 19
        False negatives: 31
        True negatives: 47
        
    Cross Validation Performance results:
    [0.7037037  0.75925926 0.74766355 0.82242991 0.74766355]
    Cross Validation mean: 0.7561439944617515 accuracy with a standard deviation of 0.038203002339734664
    Train data accuracy:  0.76909
    Test data accuracy:  0.78355
        

model 2. 
    DecisionTreeClassifier model has an Accuracy output of: 0.7272727272727273
    
    Confusion Matrix of DecisionTreeClassifier is:
                                        [[120  33]
                                        [ 30  48]]
        True positives: 120
        False positives: 33
        False negatives: 30
        True negatives: 48
        
    Cross Validation Performance results: 
        [0.68518519 0.65740741 0.6635514  0.76635514 0.6728972 ]
        Cross Validation mean: 0.6890792661820699 accuracy with a standard deviation of 0.03975767209370288
        Train data accuracy:  1.0
        Test data accuracy:  0.72727
        

model 3. 
    GaussianNB model has an Accuracy output of: 0.7748917748917749
    
    Confusion Matrix of DecisionTreeClassifier is:
                                        [[130  23]
                                        [ 29  49]]
        True positives: 130
        False positives: 23
        False negatives: 29
        True negatives: 49
        
    Cross Validation Performance results: 
        [0.68518519 0.73148148 0.71028037 0.80373832 0.80373832]
        Cross Validation mean: 0.7468847352024921 accuracy with a standard deviation of 0.04867983908653887
        Train data accuracy:  0.7635
        Test data accuracy:  0.77489

    
"""




"""
Kernel = rbf
Kernel Decision Tree = gini


model 1. 
    SVC model has an Accuracy output of: 0.7532467532467533

    Confusion Matrix of SVC model is: 
                                        [[138  18]
                                        [ 39  36]]
        True positives: 138
        False positives: 18
        False negatives: 39
        True negatives: 36
    
    Cross Validation Performance results: 
        [0.84259259 0.75       0.71962617 0.75700935 0.74766355]
        Cross Validation mean: 0.7633783316026307 accuracy with a standard deviation of 0.04160684798906006
        Train data accuracy:  0.77281
        Test data accuracy:  0.75325


model 2. 
    DecisionTreeClassifier model has an Accuracy output of: 0.7142857142857143

    Confusion Matrix of DecisionTreeClassifier is:
                                        [[115  41]
                                        [ 25  50]]
        True positives: 115
        False positives: 41
        False negatives: 25
        True negatives: 50
        
    Cross Validation Performance results: 
        [0.74074074 0.64814815 0.64485981 0.75700935 0.71028037]
        Cross Validation mean: 0.7002076843198337 accuracy with a standard deviation of 0.04635601069943756
        Train data accuracy:  1.0
        Test data accuracy:  0.71429


model 3. 
    GaussianNB model has an Accuracy output of: 0.7575757575757576

    Confusion Matrix of DecisionTreeClassifier is:
                                        [[134  22]
                                        [ 34  41]]
        True positives: 134
        False positives: 22
        False negatives: 34
        True negatives: 41
        
    Cross Validation Performance results: 
        [0.78703704 0.74074074 0.73831776 0.81308411 0.70093458]
        Cross Validation mean: 0.7560228452751817 accuracy with a standard deviation of 0.03949581793116758
        Train data accuracy:  0.76164
        Test data accuracy:  0.75758


"""








"""
Kernel SVC = Linear
Kernel Decision Tree = Entropy

Results: 


model 1. 
    SVC model has an Accuracy output of: 0.7835497835497836
    
    Confusion Matrix of SVC model is: 
                                        [[134  19]
                                        [ 31  47]]
        True positives: 134
        False positives: 19
        False negatives: 31
        True negatives: 47
        
    Cross Validation Performance results:
    [0.7037037  0.75925926 0.74766355 0.82242991 0.74766355]
    Cross Validation mean: 0.7561439944617515 accuracy with a standard deviation of 0.038203002339734664
    Train data accuracy:  0.76909
    Test data accuracy:  0.78355
        

model 2. 
    DecisionTreeClassifier model has an Accuracy output of: 0.7272727272727273
    
    Confusion Matrix of DecisionTreeClassifier is:
                                        [[120  33]
                                        [ 30  48]]
        True positives: 120
        False positives: 33
        False negatives: 30
        True negatives: 48
        
    Cross Validation Performance results: 
        [0.68518519 0.65740741 0.6635514  0.76635514 0.6728972 ]
        Cross Validation mean: 0.6890792661820699 accuracy with a standard deviation of 0.03975767209370288
        Train data accuracy:  1.0
        Test data accuracy:  0.72727
        

model 3. 
    GaussianNB model has an Accuracy output of: 0.7748917748917749
    
    Confusion Matrix of DecisionTreeClassifier is:
                                        [[130  23]
                                        [ 29  49]]
        True positives: 130
        False positives: 23
        False negatives: 29
        True negatives: 49
        
    Cross Validation Performance results: 
        [0.68518519 0.73148148 0.71028037 0.80373832 0.80373832]
        Cross Validation mean: 0.7468847352024921 accuracy with a standard deviation of 0.04867983908653887
        Train data accuracy:  0.7635
        Test data accuracy:  0.77489

    
"""




"""
Kernel = rbf
Kernel Decision Tree = gini


model 1. 
    SVC model has an Accuracy output of: 0.7532467532467533

    Confusion Matrix of SVC model is: 
                                        [[138  18]
                                        [ 39  36]]
        True positives: 138
        False positives: 18
        False negatives: 39
        True negatives: 36
    
    Cross Validation Performance results: 
        [0.84259259 0.75       0.71962617 0.75700935 0.74766355]
        Cross Validation mean: 0.7633783316026307 accuracy with a standard deviation of 0.04160684798906006
        Train data accuracy:  0.77281
        Test data accuracy:  0.75325


model 2. 
    DecisionTreeClassifier model has an Accuracy output of: 0.7142857142857143

    Confusion Matrix of DecisionTreeClassifier is:
                                        [[115  41]
                                        [ 25  50]]
        True positives: 115
        False positives: 41
        False negatives: 25
        True negatives: 50
        
    Cross Validation Performance results: 
        [0.74074074 0.64814815 0.64485981 0.75700935 0.71028037]
        Cross Validation mean: 0.7002076843198337 accuracy with a standard deviation of 0.04635601069943756
        Train data accuracy:  1.0
        Test data accuracy:  0.71429


model 3. 
    GaussianNB model has an Accuracy output of: 0.7575757575757576

    Confusion Matrix of DecisionTreeClassifier is:
                                        [[134  22]
                                        [ 34  41]]
        True positives: 134
        False positives: 22
        False negatives: 34
        True negatives: 41
        
    Cross Validation Performance results: 
        [0.78703704 0.74074074 0.73831776 0.81308411 0.70093458]
        Cross Validation mean: 0.7560228452751817 accuracy with a standard deviation of 0.03949581793116758
        Train data accuracy:  0.76164
        Test data accuracy:  0.75758


"""







"""
Comparison results::

Fra hva jeg kan se, og har erfaring gjennom oppgaven som gjelder Classification, 
kan jeg se at modellen som har hatt best accuracy score basert på mange kjøringer er: 


SVC linear:
    SVC model has an Accuracy output of: 0.7835497835497836
    Train data accuracy:  0.76909
    Test data accuracy:  0.78355

                                                                    best accuracy based on kernel = 0.78 with kernel = linear

SVC rbf:
    SVC model has an Accuracy output of: 0.7532467532467533
    Train data accuracy:  0.77281
    Test data accuracy:  0.75325




Decision Tree kernel = Entropy:
    DecisionTreeClassifier model has an Accuracy output of: 0.7272727272727273
    Train data accuracy:  1.0
    Test data accuracy:  0.72727

                                                                    Best accueacy based on criterion = 0.727 with Entropy, low difference tho..

Decision Tree kernel = gini:
    DecisionTreeClassifier model has an Accuracy output of: 0.7142857142857143
    Train data accuracy:  1.0
    Test data accuracy:  0.71429




Gaussian Naive Bayes first run: 
    GaussianNB model has an Accuracy output of: 0.7748917748917749
    Train data accuracy:  0.7635
    Test data accuracy:  0.77489

                                                                    Best accuracy on Naive bayes Gaussian = 0.77 first run. 

Gaussian Naive Bayes second run: 
    GaussianNB model has an Accuracy output of: 0.7575757575757576
    Train data accuracy:  0.76164
    Test data accuracy:  0.75758




Results show that the best accuracy for this dataset is based on the SVC "Supported Vector Machines algorith", with an accuracy on 78.3% chance on detecting diabetes on pasients. 

"""