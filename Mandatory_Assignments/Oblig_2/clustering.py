from numpy import unique
from sklearn import metrics
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import rand_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.metrics import rand_score


iris = load_iris(as_frame=True)
iris_data = iris["data"]
iris_target = iris["target"]
print("Length of x and y dataset: \n", len(iris_data), len(iris_target))

# Se første 5 radene i datasettet & info
print(iris_data.head())
print(iris_data.describe())

# Sjekke om radene inneholder forskjellige verdier.
missing_props = iris_data.isna()
print(missing_props) # output = False ingen verdier mangler

# Normalisere
iris_data = ((iris_data - iris_data.min() ) / (iris_data.max() - iris_data.min()))

# Split data into training and test data using stratified sampling; With ration 80% Train and 20% test
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.2, stratify=iris['target'])

# Exploratory data analysis using pairplot and libraries
sns.pairplot(iris_data)
#plt.show()

"""
elbow_visual = KMeans(random_state=42)

elb_visualizer = KElbowVisualizer(elbow_visual, k=(2, 11))
elb_visualizer.fit(iris_data)
elb_visualizer.show()

Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(iris_data)
    kmeans.fit(iris_data)
    Error.append(kmeans.inertia_)

plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()

# Check boxplot of dataset
sns.boxplot( data=iris_data)
plt.show()
"""

# tester ut k = x
# k = 2-4 sa første modell oss
# k = 4 sa modell nr 2 oss
# k = 4 sa også siste modell oss, men den ga oss pekepinne på at noe var galt.
# k = 3 er riktig.

# KElbows Visualizer:
KElbows_model = KMeans(random_state=42)
KElbow_visualizer = KElbowVisualizer(KElbows_model, k=(2,11))
KElbow_visualizer.fit(iris_data)
#KElbow_visualizer.show()

# Silhouettevisualizer:
solhouette_model = KMeans(n_clusters = 3, random_state=42)  # Virker som at alle eksempler og studenter bruker random:state = 42, så går for dette jeg også. er dette et symbolsk tall? heheh.
solhouette_model_visualizer = SilhouetteVisualizer(solhouette_model)
solhouette_model_visualizer.fit(iris_data)
#solhouette_model_visualizer.show()

k_means = KMeans(n_clusters=3)
k_means.fit(iris_data)


# Silhouette score
kmeans_silhouette_score = silhouette_score(iris_data, k_means.labels_)
print("K-Means Silhouette score: \n", kmeans_silhouette_score)

# RI score
kmeans_rand = rand_score(k_means.predict(x_train), y_train)
print("K-Means Rand score: \n", kmeans_rand)

# Now we can plot the results for analysis studying
labels = k_means.predict(iris_data)
x = iris_data.assign(predicted_label=labels)
#sns.pairplot(x, hue='predicted_label', height=2, markers=['D', 'o', 's'])
#plt.show()


print("\n\n\n")


# === Agglomerative clustering ===
# k = 3, fant vi ut tidligere
# linkage kan være "complete" eller "consequently", men sisnevnte gir rare resultater hver gang. så jeg gikk for "complete".

AggloClassifier = AgglomerativeClustering(n_clusters = 3, linkage = 'single', affinity="l2")
# Console output:
# Agglomerative score:
#   0.5323532834969982

clusters = AggloClassifier.fit_predict(iris_data)
agglo_score = silhouette_score(iris_data, AggloClassifier.labels_)
print("Agglomerative score: \n", agglo_score)

# RI / Rand Index score
agglo_RI = rand_score(AggloClassifier.labels_, iris_target)
print(f"Agglomerative rand score: \n", agglo_RI)

print("\n\n\n")

# == BIRCH ==
Birch_model = Birch(threshold=0.01, n_clusters=3, copy=True, compute_labels=True)
Birch_model.fit(iris_data)
birch_prediction = Birch_model.predict(iris_data)
clusters_br = unique(birch_prediction)
print("Clusters of Birch", clusters_br)
labels_br = Birch_model.labels_

score_br = metrics.silhouette_score(iris_data, labels_br)
print("Score of Birch = ", score_br)

# Plot the results of BIRCH algorithm
x = iris_data.assign(predicted_label=Birch_model.labels_)
birch_plot = sns.pairplot(x, hue='predicted_label', markers=['D', 'o', 's'])
birch_plot.fig.suptitle("BIRCH results", y=1)
plt.show()

