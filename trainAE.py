# @title Training

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import shutil
import os

from glob import glob
from os import path
from scipy import io

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import re
import pandas as pd


def normAct(activity):
    # Always chooce an odd number for this.
    WIN_SIZE = 5
    signalLength = activity.shape[1]

    pad = (np.ones((1, (WIN_SIZE - 1))))
    newActivity = np.concatenate((pad * activity[0, 0],
                                  activity,
                                  pad * activity[0, -1]),
                                 axis=1)

    newActivity = np.convolve(np.ravel(newActivity),
                              np.ravel(np.ones((1, WIN_SIZE)) / WIN_SIZE),
                              mode="valid")

    newActivity = newActivity[(WIN_SIZE - 1):len(newActivity)]
    newActivity = np.reshape(newActivity, (1, -1))

    return (newActivity)


def cnn_model_fn(traj, trajLength, latentDim):
    vaeBeta = 1
    input_layer = tf.reshape(traj, [-1, trajLength, 1, 1])

    conv1e = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 1],
        padding="same",
        activation=tf.nn.relu)

    conv2e = tf.layers.conv2d(
        inputs=conv1e,
        filters=64,
        kernel_size=[4, 1],
        padding="same",
        activation=tf.nn.relu)

    lastFilters = 128
    conv3e = tf.layers.conv2d(
        inputs=conv2e,
        filters=lastFilters,
        kernel_size=[5, 1],
        padding="same",
        activation=tf.nn.relu)

    flatConv3e = tf.reshape(conv3e, [-1, lastFilters * trajLength])

    # Now we define the latent layer. Linear activation.
    latentDense_Mus = tf.layers.dense(inputs=flatConv3e, units=latentDim)
    latentDense_Sigmas = tf.layers.dense(inputs=flatConv3e, units=latentDim)

    eps = tf.random_normal(
        shape=tf.shape(latentDense_Sigmas),
        mean=0, stddev=1, dtype=tf.float64)

    latentDense = latentDense_Mus + tf.sqrt(tf.exp(latentDense_Sigmas)) * eps

    # Another dense.
    dense2 = tf.layers.dense(inputs=latentDense,
                             units=trajLength * 32,
                             activation=tf.nn.relu)

    dense2_flat = tf.reshape(dense2, [-1, trajLength, 1, 32])

    # Decoder Conv
    conv1d = tf.layers.conv2d(
        inputs=dense2_flat,
        filters=64,
        kernel_size=[3, 1],
        padding="same",
        activation=tf.nn.relu)

    conv2d = tf.layers.conv2d(
        inputs=conv1d,
        filters=128,
        kernel_size=[4, 1],
        padding="same",
        activation=tf.nn.relu)

    output_conv = tf.layers.conv2d(
        inputs=conv2d,
        filters=1,
        kernel_size=[6, 1],
        padding="same",
        activation=tf.nn.relu)

    output_conv_flat = tf.reshape(output_conv, [-1, trajLength])
    output_dense = tf.layers.dense(inputs=output_conv_flat, units=trajLength)

    # the output layer.
    output_flat_for_conv = tf.reshape(output_dense, [-1, trajLength, 1, 1])

    outputConv = tf.layers.conv2d(
        inputs=output_flat_for_conv,
        filters=1,
        kernel_size=[4, 2],
        padding="same")

    output = tf.reshape(outputConv, [-1, trajLength])

    # The reconstruction loss is per trajectory step.
    reconstructionLoss = tf.reduce_mean(tf.norm(tf.subtract(output, traj), axis=1,
                                                ord=2))

    # Latent loss
    # KL divergence: measure the difference between two distributions
    # Here we measure the divergence between
    # the latent distribution and N(0, 1)
    latentLoss = -0.5 * tf.reduce_sum(
        1 + latentDense_Sigmas - tf.square(latentDense_Mus) -
        tf.exp(latentDense_Sigmas), axis=1)
    latentLoss = tf.reduce_mean(latentLoss)
    latentLoss = latentLoss * vaeBeta

    loss = (600 * reconstructionLoss + latentLoss)

    return (loss, reconstructionLoss, latentLoss, output, latentDense, conv1e)


def getBatch(batchDir, batchNum, trajLength):
    matFiles = glob(path.join(batchDir, '*.mat'))

    # Chosing the numbers of mats of batch
    chosenFiles = np.random.choice(matFiles, batchNum)

    trajs = np.array([])

    for fileName in chosenFiles:
        currentMat = io.loadmat(fileName)

        currentTraj = currentMat[list(currentMat.keys())[3]][0, 0:trajLength]
        currentTraj = np.reshape(currentTraj, (1, trajLength))
        currentTraj = normAct(currentTraj)

        if np.any(np.isnan(currentTraj)):
            # print('Got NaN train example.')
            continue

        if (np.any(np.isnan(currentTraj))):
            continue

        if (len(trajs) != 0):
            trajs = (np.concatenate((trajs, currentTraj), axis=0))
        else:
            trajs = currentTraj

    return trajs


def main():
    # Setting the seed
    # np.random.seed(15574)

    TRAJ_LENGTH = 120
    DATA_DIR = './TrainData/'
    LATENT_DIM = 4
    BATCH_SIZE = 200
    OUTPUT_PATH = './outputChristian/'
    # N = 200
    N = 10
    RESTORE = False

    tf.reset_default_graph()
    removeBadFiles(DATA_DIR, TRAJ_LENGTH)

    traj_ = tf.placeholder(tf.float64, [None, TRAJ_LENGTH])
    loss, reconstructionLoss, latentLoss, output, latentDense, conv1e = cnn_model_fn(traj_, TRAJ_LENGTH, LATENT_DIM)

    # Solver
    solver = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session() as sess:

        saver = tf.train.Saver()
        if (RESTORE == False):
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, './ChristianNetwork')

        for i in range(N):
            currentBatch = getBatch(DATA_DIR, BATCH_SIZE, TRAJ_LENGTH)

            sess.run(solver, feed_dict={traj_: currentBatch})
            lossValue = loss.eval(feed_dict={traj_: currentBatch});
            reconstructionLossValue = reconstructionLoss.eval(feed_dict={traj_: currentBatch});
            latentLossValue = latentLoss.eval(feed_dict={traj_: currentBatch});

            if (i % 5000 == 0):
                saveResults(latentDense, DATA_DIR, TRAJ_LENGTH, traj_)

            if (i % 100 == 0):
                print('\r' + str(i) + ". Loss: " + str(lossValue) + " Rec. Loss: " + str(reconstructionLossValue) +
                      " Latent Loss: " + str(latentLossValue) + " Latent Dim: " + str(LATENT_DIM), end="")

                fig, ax = plt.subplots(nrows=1, ncols=2)

                outputR = output.eval(feed_dict={traj_: currentBatch})
                outputR = np.reshape(outputR[1, :], (TRAJ_LENGTH, 1))

                inputR = np.reshape(currentBatch[1, :], (TRAJ_LENGTH, 1))

                plt.axes(ax[0])
                plt.plot(range(outputR.shape[0]), outputR, color='red')
                plt.xlabel('Time')
                plt.ylabel('Activation')
                plt.title('Output')
                plt.axes(ax[1])
                plt.plot(range(outputR.shape[0]), inputR)
                plt.xlabel('Time')
                plt.ylabel('Activation')
                plt.title('Input')
                plt.tight_layout()

                plt.savefig(path.join(OUTPUT_PATH, str(i) + "_" +
                                      str(lossValue)[0:8] + ".png"))
                plt.close(fig)

                saver.save(sess, './ChristianNetwork/ChristianNetwork')

        saveResults(latentDense, DATA_DIR, TRAJ_LENGTH, traj_)


def removeBadFiles(dataDir, trajLength):
    import os

    # Plotting the cluisters
    matFiles = glob(path.join(dataDir, '*.mat'))
    badFiles = 0

    numOfFiles = len(matFiles)
    for i in range(numOfFiles):
        fileName = matFiles[i]

        currentMat = io.loadmat(fileName)
        dataName = list(currentMat.keys())[3]
        currentTraj = np.reshape(currentMat[dataName][0, 0:trajLength], (1, trajLength))

        if (np.any(np.isnan(currentTraj))):
            badFiles = badFiles + 1
            os.remove(fileName)

    print('Removed ' + str(badFiles) + " \ " + str(numOfFiles))


def saveResults(latentDense, dataDir, trajLength, traj_):
    # Plotting the cluisters
    matFiles = glob(path.join(dataDir, '*.mat'))

    allTrajs = []
    allLatents = []
    allProperties = []
    allNames = []

    numOfFiles = len(matFiles)
    for i in range(numOfFiles):
        fileName = matFiles[i]

        currentMat = io.loadmat(fileName)
        dataName = os.path.basename(fileName)
        dataName = dataName.split(".")[0]

        currentTraj = np.reshape(currentMat['act'][0, 0:trajLength], (1, trajLength))

        if (np.any(np.isnan(currentTraj))):
            continue

        currentTraj = normAct(currentTraj)

        #currentProperties = dataName.split("_")
        currentProperties = list(re.match('{(.+)}_{(.+)}_{(.+)}',dataName).groups())

        allNames.append(dataName)

        currentProperties[1] = re.sub('\d', '', currentProperties[1])

        currentLatent = latentDense.eval(feed_dict={traj_: np.reshape(currentTraj, (1, trajLength))})

        allTrajs.append(currentTraj)
        allLatents.append(currentLatent)
        allProperties.append(currentProperties)

    allLatents = np.asarray(allLatents)
    allLatents = np.squeeze(allLatents)

    allProperties = np.squeeze(np.asarray(allProperties))

    # d = {'Name': allNames,
    #     'Strain': allProperties[:, 0],
    #     'Chem': allProperties[:, 1],
    #     'Neuron': allProperties[:, 2],
    #     'Step': allProperties[:, 3],
    #     'LatNeuron1': allLatents[:, 0],
    #     'LatNeuron2': allLatents[:, 1],
    #     'LatNeuron3': allLatents[:, 2]}

    d = {'Name': allNames,
         'Neuron': allProperties[:, 0],
         'Cond': allProperties[:, 1]}

    [d.update({'LatentVar' + str(i + 1): allLatents[:, i]}) for i in range(allLatents.shape[1])]

    df = pd.DataFrame(data=d)
    df_plot = df.drop(['Name', 'Neuron', 'Cond'], axis=1)

    df.to_csv('gdrive/My Drive/EddyOutput_small.csv')
    # files.download('EddyOutput_small.csv')

    shutil.make_archive(os.path.join('gdrive/My Drive/ChrisitianOutput', 'zip', './outputChristian')
    # files.download('outputData.zip')

    # import seaborn as sns
    # sns.clustermap(df_plot)


main()
