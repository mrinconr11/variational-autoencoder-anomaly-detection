import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D
from latte.functional.disentanglement.mutual_info import mig
from functions import rem_scatt, preprocess_test, preprocess
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Reshape, UpSampling2D, Lambda, GaussianNoise
from sklearn import metrics
from keras.callbacks import History
history = History()
from sklearn.cluster import KMeans

## Import train and test dataset. Include the excitation-emission matrices (EEM)
data = pd.read_csv(r'Lake_Train.csv')
test = pd.read_csv(r'Lake_Test.csv')
df = pd.DataFrame(data)
dft = pd.DataFrame(test)
df = df.append(dft)
train_data = np.array(df)
input_data_train = np.nan_to_num(train_data)

## Import clusters. 0 = not contaminated, 1 = contaminated with naphthenic acids (NAs).
train_clusters = pd.read_csv(r'Lake_Train_Cluster.csv')
test_clusters = pd.read_csv(r'Lake_Test_Clusters.csv')
dfc = pd.DataFrame(train_clusters)
dfct = pd.DataFrame(test_clusters)
dfc = dfc.append(dfct)
y_train = np.array(dfc)

## Clean data. Remove scattering from the EEM.
def rem_scatt(x, nex, nem):
    ## ranges of the excitation-emission spectra
    ex = range(240, 602, 2)
    em = range(250, 802, 2)
    ind = []
    bw = 12
    for i in range(0, len(ex)):
        for j in range(0, len(em)):
            # if em[j] <= (ex[i]+bw) and em[j] >= (ex[i]-bw):
            if em[j] <= (ex[i] + bw):
                ind.append((i) * nem + (j))
            # if em[j] <= (ex[i]*2+bw) and em[j] >= (ex[i]*2-bw):
            if em[j] >= (ex[i] * 2 - bw):
                ind.append((i) * nem + (j))
    ## Make scattering 0
    ind = np.reshape(ind, (1, len(ind)))
    x[:, ind] = 0.
    return x
input_data_train = rem_scatt(input_data_train, 181, 276)

#Reduce matrix size for the convolutional variational autoencoder (CVAE)
input_data_train = np.reshape(input_data_train, (-1, 181, 276))
input_data_train = input_data_train[:, 0:74, 0:146]
input_data_train = np.reshape(input_data_train, (-1, 74*146))
input_data_train, max_v, min_v = preprocess(input_data_train)
input_data_train = input_data_train.astype("float32")
train_images = np.reshape(input_data_train, (-1, 74, 146, 1))
x_train = train_images

## KL divergence loss
def kldiv_loss(betta):
    kl_loss = K.mean(-0.5 * K.sum(betta * (1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)), axis=-1))
    return kl_loss

## Reconstruction loss
def recon_loss(x, z_decoded):
    x = K.flatten(x)
    z_decoded = K.flatten(z_decoded)
    # Reconstruction loss
    xent_loss = K.mean(keras.metrics.mse(x, z_decoded))
    return xent_loss

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], encoding_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_var) * epsilon

## Latent space reduced to a 2 dim vector
encoding_dim = 2

## CVAE structure
x = keras.Input(shape=(74, 146, 1), name="encoder_input")
corrupted = Dropout(0.1)(x)
encoder = layers.Conv2D(filters=32, kernel_size=7)(corrupted)
encoder = layers.BatchNormalization()(encoder)
encoder = layers.Activation('elu')(encoder)
encoder = MaxPooling2D((2,2), padding='same')(encoder)
encoder = layers.Conv2D(filters=32, kernel_size=7)(encoder)
encoder = layers.BatchNormalization()(encoder)
encoder = layers.Activation('elu')(encoder)
encoder = layers.Conv2D(filters=32, kernel_size=7)(encoder)
encoder = layers.BatchNormalization()(encoder)
encoder = layers.Activation('elu')(encoder)
encoder = layers.Conv2D(filters=32, kernel_size=7)(encoder)
encoder = layers.BatchNormalization()(encoder)
encoder = layers.Activation('elu')(encoder)
encoder = MaxPooling2D((2,2), padding='same')(encoder)
encoder = layers.Conv2D(filters=32, kernel_size=7)(encoder)
encoder = layers.BatchNormalization()(encoder)
encoder = layers.Activation('elu')(encoder)
encoder = layers.Flatten()(encoder)

encoder = layers.Dense(10, activation="elu")(encoder)
z_mean = Dense(encoding_dim, name="z_mean")(encoder)
z_log_var = Dense(encoding_dim, name="z_log_var")(encoder)
z = Lambda(sampling)([z_mean, z_log_var])
encoder = keras.Model(x, [z_mean, z_log_var, z], name="encoder")

decoder_input = keras.Input(shape=(encoding_dim,), name="decoder_input")
decoder = layers.Dense(units=2 * 20 * 32, activation=tf.nn.relu)(decoder_input)
decoder = layers.Reshape(target_shape=(2, 20, 32))(decoder)
decoder = layers.Conv2DTranspose(filters=32, kernel_size=7)(decoder)
decoder = layers.BatchNormalization()(decoder)
decoder = layers.Activation('elu')(decoder)
decoder = UpSampling2D((2,2))(decoder)
decoder = layers.Conv2DTranspose(filters=32, kernel_size=7)(decoder)
decoder = layers.BatchNormalization()(decoder)
decoder = layers.Activation('elu')(decoder)
decoder = layers.Conv2DTranspose(filters=32, kernel_size=7)(decoder)
decoder = layers.BatchNormalization()(decoder)
decoder = layers.Activation('elu')(decoder)
decoder = layers.Conv2DTranspose(filters=32, kernel_size=7)(decoder)
decoder = layers.BatchNormalization()(decoder)
decoder = layers.Activation('elu')(decoder)
decoder = UpSampling2D((2,2))(decoder)
decoder = layers.Conv2DTranspose(filters=32, kernel_size=7)(decoder)
decoder = layers.BatchNormalization()(decoder)
decoder = layers.Activation('elu')(decoder)
decoder_output = layers.Conv2DTranspose(filters=1, kernel_size=7, strides=1, padding='same')(decoder)
decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

#VAE
outputs = decoder(encoder(x)[2]) #this gives the z output from the encoder
z_decoded_encoded = encoder(outputs) #this is the encoding of the decoding (cycled again for solving posterior collapse)

vae = keras.Model(x, outputs, name='vae')

xent_loss = recon_loss(x, outputs)
betta = [0.01]
kl_loss = kldiv_loss(betta)
vae.add_loss(xent_loss)
vae.add_loss(kl_loss)

vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
vae.add_metric(xent_loss, name='xent_loss', aggregation='mean')
vae.compile(optimizer='adam')

## Train the CVAE
history = vae.fit(x_train, epochs=500)

## Make predictions
z_varian, z_mean, z_predicted = encoder.predict(x_train)

## Calculate the mutual information gap (MIG) to evaluate disentanglement
mutual_info_gap = mig(z_predicted, y_concentrations)

## K-means to classify water samples contaminated with NAs and not contaminated.

kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10)
kmeans.fit(z_predicted)
y_predicted = kmeans.predict(z_predicted)

## Evaluate accuracy metrics
y_train = y_train.reshape(180)
accuracy = float("{:.3f}".format(metrics.rand_score(y_train, y_predicted)))

## Plot K-means classification 
h = 0.005  # point in the mesh [x_min, x_max]x[y_min, y_max].

## Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = z_predicted[:, 0].min() - 0.2, z_predicted[:, 0].max() + 0.2
y_min, y_max = z_predicted[:, 1].min() - 0.2, z_predicted[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

xx = xx.astype(np.double)
yy = yy.astype(np.double)

# Obtain labels for each point in mesh. Use last trained model.
to_predict = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
Z = kmeans.predict(to_predict)

import copy
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.binary, vmin=0.3, vmax=0.7,
    aspect="auto",
    origin="lower",
)

scatter = plt.scatter(z_predicted[:, 0], z_predicted[:, 1], c=probabi, cmap='seismic',
                      vmin=0, vmax=100, s=4)
plt.colorbar()
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='black', marker='x', s=15 ** 2, linewidths=3)
plt.xlabel("z[0]")
plt.ylabel("z[1]")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks()
plt.yticks()
plt.show()
