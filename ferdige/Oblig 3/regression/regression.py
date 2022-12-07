import numpy as np
import seaborn as sns
import sns as sns
from numpy.ma import arange
from pandas import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree



dataframe = pd.read_csv("concrete_data.csv", header=0, low_memory=False)
# Choose which columns to use
dataframe = dataframe[["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age", "Strength"]]




"""      Data Preprocessing     """
# Check basic info
# checking anomalies in the dataset:
print(f"\nFirst 5 rows of the normalized dataset: \n"
      f"{dataframe.head()}\n\n"
      f"Some general info about the dataset:\n"
      f"{dataframe.describe()}\n"
      f"Some extra info: \n"
      f"{dataframe.info}")

# Checking for missing values
missing_props = dataframe.isna().sum() # Expected output = 0
print(f"\nChecking if some rows are missing values or have wrong datatype:\n"
      f" {missing_props}") # Out was as expected.

""" Normalize values using min/max"""
dataframe = ((dataframe - dataframe.min() ) / (dataframe.max() - dataframe.min()))
print(f"\nFirst 5 rows of the normalized dataset: \n"
      f"{dataframe.head()}\n\n")



"""        Split data       """
# Select the features and the target of the dataset
features = dataframe.drop(["Strength"], axis=1)
target = dataframe["Strength"]

# Split dataset into 75/25 respectively Train and Test.
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25)

# Check the amount of each sets, using 2 different methods; Show off? ;)
print(f"\nLength of rows:\nx_train: {len(x_train)}\nx_test: {len(x_test)}\ny_train: {len(y_train)}\ny_test: {len(y_test)}")
print(f"Shape of datasets:\nx_train: {x_train.shape}\nx_test: {x_test.shape}\ny_train: {y_train.shape}\ny_test: {y_test.shape}")
# Output / console => Looks great! Train = 800, test = 200. all of the adds up perfectly :)


plot = sns.boxplot(dataframe, x="Age", y="Strength")
plt.show()


"""        Initiate Models       """
modeller = []

"""  Linear Regression  """

# Default startup
linear_regression_model = LinearRegression()
modeller.append(linear_regression_model)

sgd_Regressor = SGDRegressor(warm_start=False, verbose=2, validation_fraction=0.2, shuffle=False, penalty="l2", n_iter_no_change=3, max_iter=900, loss="squared_error", learning_rate="constant", l1_ratio=0.8, fit_intercept=True, eta0=0.03, epsilon=0.2, early_stopping=True, average=False, alpha=0.0008)
print("get help: ", sgd_Regressor.get_params())
modeller.append(sgd_Regressor)

# Tuning
alpha = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.005, 0.01, 0.05, 0.1]
max_depth = list((range(0, 150, 5)))
bools = [True, False]
epsilon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2]
eta0 = [0.01, 0.02, 0.03, 0.04, 0.05]
min_samples_split = list(range(2, 10))
l1_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
loss = ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]
learning_rate = ["constant", "optimal", "invscaling", "adaptive"]
max_iter = list(range(50, 5000, 50))
n_iter_no_change = [1, 2, 3, 4, 5]
penalty = ["l2", "l1", "elasticnet"]
validation_fraction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
verbose = [0, 2, 4, 5]

gridSearch = {
            "alpha": alpha,
            "average": bools,
            "early_stopping": bools,
            "epsilon": epsilon,
            "eta0": eta0,
            "fit_intercept": bools,
            "l1_ratio": l1_ratio,
            "learning_rate": learning_rate,
            "loss": loss,
            "max_iter": max_iter,
            "n_iter_no_change": n_iter_no_change,
            "penalty": penalty,
            "shuffle": bools,
            "validation_fraction": validation_fraction,
            "verbose": verbose,
            "warm_start": bools,
}
Linear_tuning = RandomizedSearchCV(estimator=sgd_Regressor, param_distributions=gridSearch, error_score="raise")
Linear_tuning.fit(x_train, y_train)

# Console log the generated parameter settings option
print(f"\nparameter generated: {Linear_tuning.best_params_}")
# output = parameter generated: {'splitter': 'best', 'min_weight_fraction_leaf': 0.2, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_leaf_nodes': 6, 'max_depth': 15}


"""  Decision Tree Regression  """
# Default model
decision_tree = DecisionTreeRegressor()
modeller.append(decision_tree)

"""
Plot tree

# Fit the model
decision_tree.fit(x_train, y_train)
# Generate score for training set
plt.figure(figsize=(10,10))
plot_tree(decision_tree, fontsize=10)
plt.show()
"""

# Tuned Decision Tree:
decision_tree_tuned = DecisionTreeRegressor(splitter="best", min_weight_fraction_leaf=0.1, min_samples_split=5, min_samples_leaf=1, max_leaf_nodes=12, max_depth=65, criterion="absolute_error")
modeller.append(decision_tree_tuned)
# parameter generated: : {'splitter': 'best', 'min_weight_fraction_leaf': 0.1, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_leaf_nodes': 12, 'max_depth': 65, 'criterion': 'absolute_error'}


# Initiate Tuning of Decision Tree parameter: criterion
# Criterion 1: squared_error
# Criterion 2: friedman_mse
# Criterion 3: absolute_error
# Criterion 4: poisson

# Friedman_mse
decision_tree_friedman_mse = DecisionTreeRegressor(criterion='friedman_mse')
modeller.append(decision_tree_friedman_mse)
# absolute_error
decision_tree_absolute_error = DecisionTreeRegressor(criterion='absolute_error')
modeller.append(decision_tree_absolute_error)
# poisson
decision_tree_poisson = DecisionTreeRegressor(criterion='poisson')
modeller.append(decision_tree_poisson)



# Tuning
splitter = ["best", "random"]
max_depth = list((range(0, 150, 5)))
bools = [True, False]
min_samples_split = list(range(2, 10, 1))
min_samples_leaf = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2]
min_weight_fraction_leaf = [0.1, 0.2, 0.3, 0.4, 0.5]
max_leaf_nodes = list(range(2, 20, 1))
criterion = ["poisson", "absolute_error", "friedman_mse", "squared_error"]
gridSearch = {
            "splitter": splitter,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_leaf_nodes": max_leaf_nodes,
            "criterion": criterion
}
decision_tree_tuning = RandomizedSearchCV(estimator=decision_tree, param_distributions=gridSearch)
decision_tree_tuning.fit(x_train, y_train)


# Console log the generated parameter settings option
print(f"\nparameter generated: {decision_tree_tuning.best_params_}")
# output = parameter generated: {'splitter': 'best', 'min_weight_fraction_leaf': 0.2, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_leaf_nodes': 6, 'max_depth': 15}




"""  Ridge Regression  """
ridge = Ridge()
#ridge = Ridge(fit_intercept=False, alpha=0.5, max_iter=3)
modeller.append(ridge)
# Output FØR tuning: 61.1 %
# Output ETTER tuning: 59.3 max_iter = 33, fit=false
# Output ETTER tuning: 60   max_iter = 6, fit_ = true, aplha 0.1


# Tuning
alphas_n = [0.0, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
max_iter = list((range(0, 50)))
bools = [True, False]
gridSearch = {
            "alpha": alphas_n,
            "max_iter": max_iter,
            "fit_intercept": bools,
}
ridge_tuning = RandomizedSearchCV(estimator=ridge, param_distributions=gridSearch)
ridge_tuning.fit(x_train, y_train)

# Print best params
print(f"\nBest param: {ridge_tuning.best_params_}")
# output = Best param: {'max_iter': 15, 'fit_intercept': True, 'alpha': 2.5}





"""  === Random Forest Regression === """
# Default model
randomForest = RandomForestRegressor()
modeller.append(randomForest)


# Tuning

# == The tuning ==
max_depth = list((range(0, 150, 5)))
bools = [True, False]
min_samples_split = list(range(2, 10))
criterion = ["squared_error", "absolute_error", "poisson"]
n_iter_no_change = [1, 2, 3, 4, 5]
verbose = [0, 2, 4, 5]
n_estimators = [5, 20, 50, 100]
min_samples_leaf = list(range(1, 50, 5))
min_weight_fraction_leaf = [0.1, 0.2, 0.3, 0.4, 0.5]
max_features = ["sqrt", "log2"]
max_leaf_nodes = list(range(2, 20, 1))
min_impurity_decrease = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
n_jobs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


gridSearchFR = {
            "n_estimators": n_estimators,
            "verbose": verbose,
            "warm_start": bools,
            "criterion": criterion,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "bootstrap": bools,
            "n_jobs": n_jobs,
            "max_depth": max_depth
}
randomForest_tuner = RandomizedSearchCV(estimator=randomForest, param_distributions=gridSearchFR, error_score="raise")
randomForest_tuner.fit(x_train, y_train)
# output

#parameter generated: {'warm_start': False, 'verbose': 4, 'n_jobs': 7, 'n_estimators': 100, 'min_weight_fraction_leaf': 0.4, 'min_samples_split': 9, 'min_samples_leaf': 11, 'min_impurity_decrease': 0.1, 'max_leaf_nodes': 3, 'max_features': 'log2', 'criterion': 'poisson', 'bootstrap': True}

# Console log the generated parameter settings option
print(f"\nparameter generated: {randomForest_tuner.best_params_}")



# Creating the best model
randomForest_tuner_bestModel = RandomForestRegressor(
    n_estimators=randomForest_tuner.best_params_['n_estimators'],
    max_depth=randomForest_tuner.best_params_['max_depth'],
    warm_start=randomForest_tuner.best_params_['warm_start'],
    verbose=randomForest_tuner.best_params_['verbose'],
    n_jobs=randomForest_tuner.best_params_['n_jobs'],
    min_weight_fraction_leaf=randomForest_tuner.best_params_['min_weight_fraction_leaf'],
    min_samples_split=randomForest_tuner.best_params_['min_samples_split'],
    bootstrap=randomForest_tuner.best_params_['bootstrap'],
    min_samples_leaf=randomForest_tuner.best_params_['min_samples_leaf'],
    min_impurity_decrease=randomForest_tuner.best_params_['min_impurity_decrease'],
    max_leaf_nodes=randomForest_tuner.best_params_['max_leaf_nodes'],
    max_features=randomForest_tuner.best_params_['max_features'],
    criterion=randomForest_tuner.best_params_['criterion'],

)
# Append model to modeller
modeller.append(randomForest_tuner_bestModel)


scores = {}
for model in modeller:
    desired_prediction = 1.0
    # Fit the model
    model.fit(x_train, y_train)
    # Generate score for training set
    training_score = model.score(x_train, y_train)
    # Generate score for Test set
    test_score = model.score(x_test, y_test)
    print(f"\n\n"
          f"Model used: {model}\n"
          f"Training score: {round(training_score, 3)}\n"
          f"Test score: {round(test_score, 3)}\n"
          f"Avvik mellom Train & Test set: {round((training_score - test_score), 3)}\n"
          f"Vi er {round((desired_prediction - test_score), 3)} fra {desired_prediction}\n"
          f"med et resultat på: {((round(test_score, 3) * 100))}%"
    )
    predicted_percent = (round(test_score, 3) * 100) # Predicted percentage..
    scores.update({model: round(predicted_percent, 3)}) # Adding the model and the percent into a dict.

# Print out table of the models performance Sorted:
print("\n\n{:<55} {:<10}".format("Model", "Percent"))
scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
for bestmodel in scores:
    print("{:<55} {:<10}".format(str((bestmodel)), scores[bestmodel]))

