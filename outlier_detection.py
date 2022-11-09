import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random


def outlier_detection(x_train_, y_):
    isf = IsolationForest(n_jobs=-1, random_state=1)
    isf.fit(x_train_, y_)
    pred = isf.predict(x_train_)
    # print(isf.score_samples(x_train))
    delsample = np.where(pred == -1)
    print(delsample)
    x_train_ = np.delete(x_train_, delsample, axis=0)
    y_ = np.delete(y_, delsample, axis=0)
    return x_train_, y_


def outlier_detection_gmm(x_train, y, dimensions, percentile=5, plot=False):

    # extract Principle Components
    pca = PCA(n_components=dimensions, svd_solver='full')

    # pca.fit already returns data projected on lower dimensions
    pi = pca.fit_transform(x_train)

    # Fit Gaussian Mixture Model
    gm = GaussianMixture(n_components=1, random_state=42).fit(pi)

    # compute probability of each sample belonging to the GMM
    densities = gm.score_samples(pi)

    # set a density threshold to 4th percentile of the overall density distribution
    # check (https://en.wikipedia.org/wiki/Percentile) for percentile definition
    density_threshold = np.percentile(densities, percentile)

    # remove anomalies from pi
    pi_new = pi[densities >= density_threshold]

    # plot 2D result
    if(plot==True):
        plot_gmm(gm, pi, pi_new, dimensions)

    # remove anomalies from x_train & y
    x_train = x_train[densities >= density_threshold]
    y = y[densities >= density_threshold]

    # print number of outliers
    outliers = pi[densities < density_threshold]
    # print(outliers)
    print(f"Outlier Detection: \r\n {outliers.shape[0]} outliers have been found and removed from gmm on pca")

    return x_train, y


def plot_gmm(gm, pi, pi_new, dimensions):

    # predict labels
    labels = gm.predict(pi)
    labels_new = gm.predict(pi_new)

    # generate random dimension
    list = np.arange(0,dimensions-1,1)
    d1=random.choice(list)
    list = list[list != d1]
    d2=random.choice(list)
    list = list[list != d2]
    d3=random.choice(list)
    list = list[list != d3]
    d4=random.choice(list)

    # plot Training set projected on PCA Dimensions WITHOUT outlier removal
    fig1, axs1 = plt.subplots(2, 2)
    fig1.suptitle('WITHOUT outlier removal', fontsize=16)
    axs1[0, 0].scatter(pi[:, d1], pi[:, d2], c=labels, s=40, cmap='viridis')
    axs1[0, 1].scatter(pi[:, d1], pi[:, d3], c=labels, s=40, cmap='viridis')
    axs1[1, 0].scatter(pi[:, d1], pi[:, d4], c=labels, s=40, cmap='viridis')
    axs1[1, 1].scatter(pi[:, d2], pi[:, d3], c=labels, s=40, cmap='viridis')
    plt.show()

    # plot Training set projected on PCA Dimensions WITH outlier removal
    fig2, axs2 = plt.subplots(2, 2)
    fig2.suptitle('WITH outlier removal', fontsize=16)
    axs2[0, 0].scatter(pi_new[:, d1], pi_new[:, d2], c=labels_new, s=40, cmap='viridis')
    axs2[0, 1].scatter(pi_new[:, d1], pi_new[:, d3], c=labels_new, s=40, cmap='viridis')
    axs2[1, 0].scatter(pi_new[:, d1], pi_new[:, d4], c=labels_new, s=40, cmap='viridis')
    axs2[1, 1].scatter(pi_new[:, d2], pi_new[:, d3], c=labels_new, s=40, cmap='viridis')
    plt.show()
