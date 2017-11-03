######################################################################
# Name: kmeans2d.py
# Description: Implements K-means clustering in two dimensions.
#   Animates the algorithm using matplotlib.
# Author: Najam Syed
# Date: 2017-11-02
######################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import colorsys

K = 8   # Number of centroids to compute
numClusters = 8 # The actual number of clusters to generate
ptsPerCluster = 60  # The number of points per actual cluster
varianceCoeff = 0.01    # This will control the spread (variance) of clustered points

# xCenterBounds and yCenter bounds specify the (min, max) x and y coords,
# respectively, within which to randomly place the actual cluster centers.
xCenterBounds = (-2, 2)
yCenterBounds = (-2, 2)

# Randomly choose cluster coords. Each row represents a center. Col 0 contains
# x coords, col 1 contains y coords.
centers = np.random.random_sample((numClusters,2))
centers[:,0] = centers[:,0] * (xCenterBounds[1] - xCenterBounds[0]) + xCenterBounds[0]
centers[:,1] = centers[:,1] * (yCenterBounds[1] - yCenterBounds[0]) + yCenterBounds[0]

# The array "points" will contain all data points. Each row represents one point.
# Col 0 contains the x coords, col 1 the y coords.
points = np.zeros((numClusters * ptsPerCluster, 2))
covariance = np.array([[varianceCoeff * (xCenterBounds[1] - xCenterBounds[0]), 0],
                       [0, varianceCoeff * (yCenterBounds[1] - yCenterBounds[0])]])

for i in range(numClusters):
    points[i*ptsPerCluster:(i+1)*ptsPerCluster,:] = np.random.multivariate_normal(
        centers[i,:], covariance, ptsPerCluster)

# Randomly choose K initial centroid coords within the bounds of the data.
xDataBounds = (np.amin(points[:,0]), np.amax(points[:,0]))
yDataBounds = (np.amin(points[:,1]), np.amax(points[:,1]))
centroids = np.random.random_sample((K, 2))
centroids[:,0] = centroids[:,0] * (xDataBounds[1] - xDataBounds[0]) + xDataBounds[0]
centroids[:,1] = centroids[:,1] * (yDataBounds[1] - yDataBounds[0]) + yDataBounds[0]

initialCentroids = np.copy(centroids)

# Initialize plot and related variables for each cluster and centroid.
fig, ax = plt.subplots()
centerPoints, = ax.plot(centers[:,0], centers[:,1], ls='None', marker='^', color='k')
xViewOverhang = (xDataBounds[1] - xDataBounds[0]) * 0.1
yViewOverhang = (yDataBounds[1] - yDataBounds[0]) * 0.1
ax.set_xlim(xDataBounds[0] - xViewOverhang, xDataBounds[1] + xViewOverhang)
ax.set_ylim(yDataBounds[0] - yViewOverhang, yDataBounds[1] + yViewOverhang)

iteration = 0
iterText = ax.annotate(
    "i = {:d}".format(iteration), xy=(0.01,0.01), xycoords='axes fraction')

# Generate a unique RGB color for each centroid (K-1 unique colors).
hues = np.linspace(0, 1, K+1)[:-1]
centroidColors = np.zeros((K, 3))

centroidPointsList = []
clusterPointsList = []
for k in range(K):
    clusterColor = tuple(colorsys.hsv_to_rgb(hues[k], 0.8, 0.8))

    centroidPoint, = ax.plot([], [], ls='None', marker='o', color=clusterColor)
    clusterPoints, = ax.plot([], [], ls='None', marker='x', color=clusterColor)

    centroidPointsList.append(centroidPoint)
    clusterPointsList.append(clusterPoints)

# Create a function to assign each point to its nearest centroid. Store these
# assignments in the array "pts2Centroids" as an int from 0 to K-1.
# Update the plot with each cluster of points.
pts2Centroids = np.zeros((len(points[:,0]),), dtype=np.int)
def assignPtsToCentroids():
    for i in range(len(points[:,0])):
        # Initialize smallestDistance to a value larger than the span of the
        # dataset. This ensures the initial value will not be the smallest value.
        smallestDistance = 2 * math.sqrt((xDataBounds[1] - xDataBounds[0])**2 +
            (yDataBounds[1] - yDataBounds[0])**2)
        for k in range(K):
            currentDistance = np.linalg.norm(points[i,:] - centroids[k,:])
            if currentDistance < smallestDistance:
                smallestDistance = currentDistance
                pts2Centroids[i] = k

def recalcCentroids():
    for k in range(K):
        sumCoordinates = np.array([[0, 0]], dtype=np.float)
        numPts = 0
        for j in range(len(points[:,0])):
            if pts2Centroids[j] == k:
                sumCoordinates += points[j,:]
                numPts += 1
        # If numPts == 0, set numPts = 1 to avoid divide-by-zero.
        if numPts == 0: numPts = 1
        centroids[k,:] = sumCoordinates / numPts

def updatePlot():
    global centroidPointsList, clusterPointsList, iteration, iterText
    for k in range(K):
        centroidPointsList[k].set_data(centroids[k,0], centroids[k,1])

        clusterIndices = pts2Centroids == k
        clusterPointsList[k].set_data(points[clusterIndices,0], points[clusterIndices,1])

lastCentroids = centroids + 1

def generator():
    global centroids, lastCentroids, iteration
    while np.array_equal(centroids, lastCentroids) == False:
        lastCentroids = np.copy(centroids)

        assignPtsToCentroids()
        recalcCentroids()
        iteration += 1
        yield centroids

def init():
    global centroidPointsList, clusterPointsList, iteration, iterText
    global initialCentroids, centroids, lastCentroids
    centroids = np.copy(initialCentroids)
    iteration = 0
    iterText.set_text("i = 0")
    for k in range(K):
        centroidPointsList[k].set_data([], [])
        clusterPointsList[k].set_data([], [])

    assignPtsToCentroids()
    updatePlot()
    lastCentroids = centroids + 1

def animate(frame):
    global centroidPointsList, clusterPointsList
    iterText.set_text("i = {:d}".format(iteration))
    for k in range(K):
        centroidPointsList[k].set_data(centroids[k,0], centroids[k,1])
        clusterIndices = pts2Centroids == k
        clusterPointsList[k].set_data(points[clusterIndices,0], points[clusterIndices,1])
    plt.pause(0.4)

ani = animation.FuncAnimation(fig, animate, frames=generator, 
    init_func=init, repeat=True)

plt.show()
