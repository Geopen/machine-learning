
"""
Program to perform K-means clustering on a set of generated
clusters.

Note: Centroids can take any value, not constrained to be a
data point.
"""

import numpy as np
import matplotlib.pyplot as plt


"""
Global Variables:
"""

DATA_NUM_MEANS = 4               # Number of clusters to generate data
NUM_DATA = 200                   # Number of data points generated
CENTROIDS_MIN_SEP_DIST = 2       # Minimum distance between cluster centers
CLUSTER_MAX_RAD = 1.6            # Maximum radius of generated cluster
X_MIN = -5                       # Minimum x-axis value
X_MAX = 5                        # Maximum x-axis value
Y_MIN = -5                       # Minimum y-axis value
Y_MAX = 5                        # Maximum y-axis value


"""
Data Generation:
"""

def generate_coord():
    x = np.random.uniform(X_MIN, X_MAX)
    y = np.random.uniform(Y_MIN, Y_MAX)
    return x, y

def calc_dist(vec_1, vec_2):
    x_dist = abs(vec_1[0] - vec_2[0])
    y_dist = abs(vec_1[1] - vec_2[1])
    euclid_dist = pow(pow(x_dist, 2) + pow(y_dist, 2), 0.5)
    return euclid_dist

def generate_centroids(n, s):
    # Array to store centroid co-ords:
    centroids = np.zeros([n,2])

    # Generate first centroid co-ord:
    centroids[0][0], centroids[0][1] = generate_coord()

    # Generate rest of centroid co-ords:
    for i in range(n-1):
        # Generates next centroid if too close to other generates another
        while True:
            check = 0
            centroids[i][0], centroids[i][1] = generate_coord()
            for j in range(i):
                # Centroid distance check:
                if calc_dist(centroids[j], centroids[i]) < s:
                    check = 1
            if check != 1:
                break
    return centroids

def generate_data(centroids, n):
    # Array to store data:
    data = np.zeros([n,3])

    # Generate Points around each centroid:
    for i in range(n):
        choice = np.random.choice(len(centroids[:,0]))
        r = np.random.uniform(0, CLUSTER_MAX_RAD)
        theta = np.random.uniform(0, 2) * np.pi
        data[i][0] = centroids[choice, 0] + r*np.cos(theta)
        data[i][1] = centroids[choice, 1] + r*np.sin(theta)
    return data


"""
Algorithm: 
"""

def initial_means(n, dataset):
    pred_centroids = np.zeros([n, 2])
    size = len(dataset[:,0])
    # Picks n random points from dataset to serve as initial means
    for i in range(n):
        choice = np.random.randint(size)
        pred_centroids[i][0] = dataset[choice, 0]
        pred_centroids[i][1] = dataset[choice, 1]
    return pred_centroids

def assign_clusters(centroids, dataset):
    num_data = len(dataset[:,0])
    num_centroids = len(centroids[:,0])
    count = 0

    for i in range(num_data):
        dist = np.inf
        old_cluster = dataset[i][2]
        
        for j in range(num_centroids):
            # Calculate distance from point to centroid:
            centroid_dist = calc_dist(centroids[j], dataset[i,0:2])

            # Assign point to closest centroid cluster:
            if dist > centroid_dist:
                dataset[i][2] = j
                dist = centroid_dist

        new_cluster = dataset[i][2]

        # Test if assigned cluster has changed:
        if old_cluster != new_cluster:
            count += 1
    return dataset, count

def update_centroids(centroids, dataset):
    num_data = len(dataset[:,0])
    num_centroids = len(centroids[:,0])

    # centroid_means row is [total x, total y, count]
    centroid_means = np.zeros([num_centroids, 3])

    # Accumulate clusters:
    for i in range(num_data):
        for j in range(num_centroids):
            if dataset[i][2] == j:
                centroid_means[j][0] += dataset[i][0]
                centroid_means[j][1] += dataset[i][1]
                centroid_means[j][2] += 1

    # Calculate new centroids:
    for i in range(num_centroids):
        centroids[i][0] = centroid_means[i][0] / centroid_means[i][2]
        centroids[i][1] = centroid_means[i][1] / centroid_means[i][2]
    return centroids

# Algorithm initialisation:
centroids =  generate_centroids(DATA_NUM_MEANS, CENTROIDS_MIN_SEP_DIST)
data = generate_data(centroids, NUM_DATA)
init_means = initial_means(DATA_NUM_MEANS, data)
data, count = assign_clusters(init_means, data)
centroids = update_centroids(centroids, data)

# Iterate until convergence:
while count > 0:
    data, count = assign_clusters(centroids, data)
    centroids = update_centroids(centroids, data)

# Plotting:
plt.scatter(data[:,0], data[:, 1], c = data[:,2], cmap=plt.cm.gist_rainbow)
plt.title("K-Means Clustering (K=%d)" % DATA_NUM_MEANS) 
plt.show()
