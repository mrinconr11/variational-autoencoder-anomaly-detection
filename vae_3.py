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
## train
data = pd.read_csv(r'Lake_Train_largo.csv')
test = pd.read_csv(r'Lake_Test_largo.csv')
df = pd.DataFrame(data)
dft = pd.DataFrame(test)
df = df.append(dft)
train_data = np.array(df)
input_data_train = np.nan_to_num(train_data)
##
train_clusters = pd.read_csv(r'Lake_Train_Cluster_16.csv')
test_clusters = pd.read_csv(r'Lake_Test_Clusters_16.csv')
dfc = pd.DataFrame(train_clusters)
dfct = pd.DataFrame(test_clusters)
dfc = dfc.append(dfct)
y_train = np.array(dfc)
##
train_concentration = pd.read_csv(r'Lake_Train_Concentrations_largo.csv')
test_concentration = pd.read_csv(r'Lake_Test_Concentrations_largo.csv')
dc = pd.DataFrame(train_concentration)
dct = pd.DataFrame(test_concentration)
dc = dc.append(dct)
y_concentrations = np.array(dc)
##
n_ex = 160
n_em = 240
batch_size = 32

input_data_train = rem_scatt(input_data_train, 181, 276)
#Reduce matrix size for CNN
input_data_train = np.reshape(input_data_train, (-1, 181, 276))
input_data_train = input_data_train[:, 0:74, 0:146]
input_data_train = np.reshape(input_data_train, (-1, 74*146))
input_data_train, max_v, min_v = preprocess(input_data_train)
input_data_train = input_data_train.astype("float32")
train_images = np.reshape(input_data_train, (-1, 74, 146, 1))
x_train = train_images
## Plot the clusters
def plot_label_clusters(encoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    _, _, z_predicted = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(z_predicted[:, 0], z_predicted[:, 1], c=labels, label=('none', 'NAs', 'phenol', 'both'))
    #plt.colorbar()
    plt.legend(*scatter.legend_elements(),
                        loc="upper left", title="Classes")

    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    return z_predicted
##
beta = [0.01]
alpha = [0.01]
zzrec_loss = []
zzkl_loss = []
zzmig_loss = []
zzaccuracy_loss = []
zztotal_loss = []
##
for bet in beta:
    betta = bet
    #alppha = alp
    def kldiv_loss(betta):
        kl_loss = K.mean(-0.5 * K.sum(betta * (1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)), axis=-1))
        return kl_loss

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

    encoding_dim = 2
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
    kl_loss = kldiv_loss(betta)
    vae.add_loss(xent_loss)
    vae.add_loss(kl_loss)

    vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    vae.add_metric(xent_loss, name='xent_loss', aggregation='mean')
    vae.compile(optimizer='adam')

    history = vae.fit(x_train, epochs=500)
    ##
    z_varian, z_mean, z_predicted = encoder.predict(x_train)
    ##
    #mutual_info_gap = mig(z_predicted, y_concentrations)
    ##
    #pd.DataFrame(z_predicted).to_csv('predicted_encoder.csv')
    #decoded_data = decoder.predict(z_predicted)
    #decoded_center_1 = decoder.predict([[-0.3875, -0.0775]])
    decoded_matrix = decoder.predict([
        [-0.6, 0.4],	[-0.4, 0.4],	[-0.2, 0.4],	[0.0, 0.4],	[0.2, 0.4],	[0.4, 0.4],
        [-0.6, 0.2],	[-0.4, 0.2],	[-0.2, 0.2],	[0.0, 0.2],	[0.2, 0.2],	[0.4, 0.2],
        [-0.6, 0],	[-0.4, 0],	[-0.2, 0],	[0.0, 0],	[0.2, 0],	[0.4, 0],
        [-0.6, -0.2],	[-0.4, -0.2],	[-0.2, -0.2],	[0.0, -0.2],	[0.2, -0.2],	[0.4, -0.2],
        [-0.6, -0.4],	[-0.4, -0.4],	[-0.2, -0.4],	[0.0, -0.4],	[0.2, -0.4],	[0.4, -0.4],
        [-0.6, -0.6],	[-0.4, -0.6],	[-0.2, -0.6],	[0.0, -0.6],	[0.2, -0.6],	[0.4, -0.6],
])
    excitation = np.arange(240, 388, 2)
    emission = np.arange(250, 542, 2)
    #49680
    ex = range(240, 388, 2)
    em = range(250, 542, 2)

    Y, X = np.meshgrid(emission, excitation)
    z = x_train[38]
    Z = z.reshape(Y.shape)
## Aca cambiar para probar diferentes imagenes
    def renormalize_01(x, max_v, min_v):
        y = np.copy(x)
        for i in range(np.size(max_v)):
            if (max_v[i] - min_v[i] > 0):
                y[:, i] = (x[:, i]) * (max_v[i] - min_v[i]) + min_v[i]
            else:
                continue
        return y

    fig, axes = plt.subplots(nrows=6, ncols=6)
    i = 0
    for ax in axes.flat:
        ax.set_axis_off()
        z2 = np.reshape(decoded_matrix[i], (-1, 74 * 146))
        i = i + 1
        z2 = renormalize_01(z2, max_v, min_v)
        Z2 = z2.reshape(Y.shape)
        im = ax.pcolormesh(X, Y, Z2, cmap='jet')
    cbar = fig.colorbar(im, ax=axes)
    plt.show()
    ##
    z2 = np.reshape(decoded_matrix[30], (-1, 74 * 146))
    z2 = renormalize_01(z2, max_v, min_v)
    Z2 = z2.reshape(Y.shape)
    im = plt.pcolormesh(X, Y, Z2, cmap='jet')
    cbar = plt.colorbar(im)
    plt.show()

    # reduced_data, _, _ = encoder.predict(x_train)
    # kmeans_kwargs = {
    #     "init": "random",
    #     "n_init": 10,
    #     "max_iter": 300,
    #     "random_state": 42,
    # }
    # # A list holds the SSE values for each k
    # sse = []
    # for k in range(1, 11):
    #     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    #     kmeans.fit(reduced_data)
    #     sse.append(kmeans.inertia_)
    #
    # #plt.style.use("fivethirtyeight")
    # plt.plot(range(1, 11), sse)
    # plt.xticks(range(1, 11))
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("SSE")
    # plt.show()
##
    #plot_label_clusters(encoder, x_train, probabi)
    plt.figure(figsize=(12, 10))

    kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10)
    kmeans.fit(z_predicted)
    y_predicted = kmeans.predict(z_predicted)

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='black', marker='x', s=15**2, linewidths=3)
    # draw enclosure
    from scipy.spatial import ConvexHull

    for label in set(kmeans.labels_):
        X_clust = z_predicted[kmeans.labels_ == label]
        hull = ConvexHull(X_clust, qhull_options='QJ')
        vertices_cycle = hull.vertices.tolist()
        vertices_cycle.append(hull.vertices[0])
        plt.plot(X_clust[vertices_cycle, 0], X_clust[vertices_cycle, 1], 'k--',
                 lw=1)
        plt.scatter(X_clust[:, 0], X_clust[:, 1])
    scatter = plt.scatter(z_predicted[:, 0], z_predicted[:, 1], c=probabi, cmap='jet',
                          vmin=0, vmax=100)
    plt.colorbar()

    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

    y_train = y_train.reshape(180)
    accuracy = float("{:.3f}".format(metrics.rand_score(y_train, y_predicted)))
    ##
    zlosses = float("{:.3f}".format(history.history['loss'][-1]))
    zkl_loss = float("{:.3f}".format(history.history['kl_loss'][-1]))
    zxent_loss = float("{:.3f}".format(history.history['xent_loss'][-1]))
    ##
    zzkl_loss.append(zkl_loss)
    zzrec_loss.append(zxent_loss)
    zztotal_loss.append(zlosses)
    zzaccuracy_loss.append(accuracy)
    #zzmig_loss.append(mutual_info_gap)
##
    probabi = [100,32,0,100,0,100,0,72,0,0,100,38,0,0,100,100,96,36,0,100,0,100,0,15,100,0,0,82,100,91,0,0,100,0,100,100,0,100,100,0,89,0,
               47,0,0,0,32,97,100,100,0,0,100,100,100,100,100,1,0,2,98,0,0,100,0,100,100,12,2,100,21,0,100,100,100,0,
               100,0,100,0,100,0,88,0,2,0,3,100,0,8,100,100,100,0,91,0,43,0,100,0,100,0,100,100,8,100,100,0,98,100,
              100,100,0,100,100,100,100,0,100,0,0,100,0,100,0,98,0,100,11,100,0,0,0,0,100,0,100,100,36,0,0,100,100,100,
              93,100,59,0,100,100,0,99,1,100,100,0,1,100,100,0,100,100,0,0,0,0,100,0,100,0,100,0,100,100,0,0,99,100,85,60]
    ax = plt.axes(projection='3d')
    #plt.figure(figsize=(12, 10))
    scatter = ax.scatter3D(z_predicted[:, 0], z_predicted[:, 1], probabi, c=y_train)

    #ax.colorbar()

    #ax.legend(*scatter.legend_elements(),
    #                    loc="upper left", title="Classes")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_zlabel('probability')
    plt.show()
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
    ##
    y_predicted_array = []
    accuracy_array = []
    for x in range(100):
        z_varian, z_mean, z_predicted = encoder.predict(x_train)

        y_predicted = kmeans.predict(z_predicted)
        #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
        #            c='black', marker='x', s=15**2, linewidths=3)
        #plt.show()
        y_predicted_array.append(y_predicted)
        y_train = y_train.reshape(180)
        accuracy = float("{:.3f}".format(metrics.rand_score(y_train, y_predicted)))
        accuracy_array.append(accuracy)
        ##
    datafr = pd.DataFrame(z_predicted)
    datafr.to_csv('paper3_zPredicted.csv')
        #zzaccuracy_loss.append(y_predicted)
##
dfaccurazy = pd.DataFrame(zzaccuracy_loss)

# saving the dataframe
dfaccurazy.to_csv('GFG.csv')
y = 0
##
h = 0.005  # point in the mesh [x_min, x_max]x[y_min, y_max].
# Plot the decision boundary. For that, we will assign a color to each
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