
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from sklearn.neural_network import MLPClassifier
import seaborn as sns


# import dataset for diabetes
diabetes = pd.read_csv("diabetes.csv")
diabetes = ((diabetes - diabetes.min()) / (diabetes.max() - diabetes.min()))

# show firs 5 rows in dataset:
print(diabetes.head())
# Check dataset for values, if non-null etc..
print(diabetes.info())

# Target the features and target
# Assign targets and features
target = diabetes['Outcome']
feature = diabetes.drop('Outcome', axis=1)

# Following teacher to use onehot encoding
# Target
target_encoded = pd.get_dummies(target)
target_encoded = target_encoded.astype('int')

# feature
feature_encoded = pd.get_dummies(diabetes)
feature_encoded = feature_encoded.astype('float64')


# Split dataset into 75/25 Train - Test sets.
x_train, x_test, y_train, y_test = train_test_split(feature_encoded, target_encoded, test_size=0.25, stratify=target)

# check the shape
print("\n\n{:<10} {:<10}".format("x shape", "y shape"))
print("{:<10} {:<10}".format(x_train.shape[0], y_train.shape[0]))
# Output = 576 for both.


"""
Testa ut en annen type som ikke funka:

classifier = Sequential()
# Defining the Input layer and FIRST hidden layer,both are same!
# relu means Rectifier linear unit function
classifier.add(Dense(units=10, input_dim=9, kernel_initializer='uniform', activation='relu'))

#Defining the SECOND hidden layer, here we have not defined input because it is
# second layer and it will get input as the output of first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# And output_dim will be equal to the number of factor levels
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='SGD()', loss='categorical_crossentropy', metrics=['accuracy'])
"""

# convert to tensorflow
train_features_tensor = tf.convert_to_tensor(x_train)
test_features_tensor = tf.convert_to_tensor(x_test)
train_targets_tensor = tf.convert_to_tensor(y_train)
test_targets_tensor = tf.convert_to_tensor(y_test)


# make the model
model = Sequential([
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.2),
    Dense(25, activation='sigmoid'),
    Dense(2, activation='sigmoid')
])
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(train_features_tensor, train_targets_tensor, epochs=10, batch_size=32)

# Score the model
print(model.evaluate(test_features_tensor, test_targets_tensor))


""" ==== OBS ====
    Her har jeg fått litt hjelp av Andreas Mathisen, 
    inspirasjon og hjelp da jeg stod fast.. (under her)
"""


# Tuning
def create_model(drop_out):
    model = Sequential([
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(drop_out),
        Dense(25, activation='sigmoid'),
        Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create a classifier for the given model above
classifier = KerasClassifier(create_model, epochs=10, batch_size=32)

# Gridsearch for finding best parameters to use:
params = {'drop_out': [0.1, 0.2, 0.3, 0.4]}
grid = GridSearchCV(classifier, params)
grid.fit(x_train, y_train)

"""
plt.figure(1)
grid_results = pd.DataFrame(grid.cv_results_)
plt.plot(grid_results['param_drop_out'], grid_results['mean_test_score'])
plt.show()
"""

# Create last model
final_model = create_model(grid.best_params_['drop_out'])
final_model.fit(x_train, y_train)
print(f"Final evaluation: \n"
      f"{final_model.evaluate(x_train, y_train)}\n\n")



"""         Create Multi Layer Perception           """
max_iter = len(x_test)
modeller = []

# Default model
multi_layer_model_1 = MLPClassifier()
modeller.append(multi_layer_model_1)



# Tuning
# == The tuning ==
activation = ["identity", "logistic", "tanh", "relu"]
solver = ["lbfgs", "sgd", "adam"]
batch_size = list(range(1, 5, 1))
learning_rate = ["constant", "invscaling", "adaptive"]

gridSearch_Multi_Layer = {
            "activation": activation,
            "solver": solver,
            "batch_size": batch_size,
            "learning_rate": learning_rate,

}
mlpc_tuner = RandomizedSearchCV(
            estimator=multi_layer_model_1,
            param_distributions=gridSearch_Multi_Layer,
            error_score="raise")
mlpc_tuner.fit(x_train, y_train)


# Console log the generated parameter settings option
print(f"\nparameter generated: {mlpc_tuner.best_params_}")



# Creating the best model
multi_layer_tuner = MLPClassifier(
    activation=mlpc_tuner.best_params_['activation'],
    solver=mlpc_tuner.best_params_['solver'],
    batch_size=mlpc_tuner.best_params_['batch_size'],
    learning_rate=mlpc_tuner.best_params_['learning_rate'],
)
# Append model to modeller
modeller.append(multi_layer_tuner)


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
print("\n\n{:<65} {:<10}".format("Model", "Percent"))
scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
for bestmodel in scores:
    print("{:<65} {:<10}".format(str((bestmodel)), scores[bestmodel]))




