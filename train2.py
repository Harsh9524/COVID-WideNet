import h5py
import glob
import random
import matplotlib
import numpy as np
from keras import optimizers
from keras import backend as K
from skimage import transform
from keras import models, layers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
matplotlib.use('Agg')
from matplotlib import pyplot as plt


train_data = np.load("x_train.npy")
train_data = np.expand_dims(train_data, axis=3)
y_train = np.load("y_train.npy")
train_labels = to_categorical(y_train, num_classes=2)

def build_model(calc_margin):
    print("Building model...")
    number_of_classes = 2
    input_shape = (64,64, 1)

    x = layers.Input(shape=input_shape)
    '''
    Inputs to the model are MRI images which are down-sampled
    to 64 × 64 from 512 × 512, in order to reduce the number of
    parameters in the model and decrease the training time.
    Second (First?) layer is a convolutional layer with 64 × 9 × 9 filters
    and stride of 1 which leads to 64 feature maps of size 56×56.
    '''
    conv1 = layers.Conv2D(64, (3, 3), activation='relu',
                          name="FirstLayer")(x)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu',
                          name="SecondLayer")(x) 
    conv3 = layers.Conv2D(256, (3, 3), activation='relu',
                          name="ThirdLayer")(x)                         
    '''
    The second layer is a Primary Capsule layer resulting from
    256×9×9 convolutions with strides of 2.
    This layer consists of 32 capsules with dimension of 8 each of
    which has feature maps of size 24×24 (i.e., each Component
    Capsule contains 24 × 24 localized individual Capsules).
    '''
    primaryCaps = PrimaryCap(inputs=conv3, dim_capsule=8,
                             n_channels=32, kernel_size=9, strides=2,
                             padding='valid')
    '''
    Final capsule layer includes 3 capsules, referred to as “Class
    Capsules,’ ’one for each type of candidate brain tumor. The
    dimension of these capsules is 16.
    '''
    capLayer2 = CapsuleLayer(num_capsule=2, dim_capsule=16, routings=3,
                             name="ThirdLayer")(primaryCaps)

    out_caps = Length(name='capsnet')(capLayer2)

    # Decoder network.
    y = layers.Input(shape=(number_of_classes,))
    # The true label is used to mask the output of capsule layer. For training
    masked_by_y = Mask()([capLayer2, y])

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu',
                             input_dim=16 * number_of_classes))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])

    if calc_margin is True:
        loss_func = [margin_loss, 'mse']
    else:
        loss_func = ['mse']
    opt = optimizers.Adam(lr=0.01)
    # model.compile(loss='categorical_crossentropy', optimizer=opt)
    train_model.compile(optimizer=opt, loss=loss_func,
                        metrics=['accuracy'])

    return train_model

def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred))
    L += 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L))

def create_generator(train_data, train_labels, batch):
    train_datagen = ImageDataGenerator()
    generator = train_datagen.flow(train_data, train_labels,
                                   batch_size=batch)
    while 1:
        x_batch, y_batch = generator.next()
        # print("y_batch", y_batch)
        yield ([x_batch, y_batch], [y_batch, x_batch])

batch = 32
def train(train_data, train_labels, num_epoch, use_margin):
    results = []
    batch = 32
    checkpointer = ModelCheckpoint(filepath='CapsNet.h5',
                                   monitor='val_capsnet_acc', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_capsnet_acc', patience=4)
    model = build_model(use_margin)
    print(model.summary(use_margin))
    val_data = np.load("x_valid.npy")
    val_data = np.expand_dims(val_data, axis=3)
    y_valid = np.load("y_valid.npy")
    val_labels = to_categorical(y_valid, num_classes=2)
    print("Training data shape: {}".format(train_data.shape))
    print("Training Labels shape: {}".format(train_labels.shape))
    print("Validation data shape: {}".format(val_data.shape))
    print("Validation Labels shape: {}".format(val_labels.shape))
    train_gen = create_generator(train_data,
                                train_labels,
                                batch)
    steps_per_epoch = len(train_data) // batch
    hst = model.fit_generator(train_gen,
                              validation_data=([val_data, val_labels],
                              [val_labels, val_data]),
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=len(val_data) // batch,
                              epochs=num_epoch,
                              verbose=1,
                              callbacks=[checkpointer,
                              early_stopping])
    results.append(hst.history)
    return results

def main():
    # full_image = False
    # sep_conv = False
    use_margin_loss = True
    # num_folds = 5
    epochs = 15
    # train_data, train_labels = prepare_data(full_image)
    train_model = train(train_data, train_labels, epochs,use_margin_loss)





if __name__ == '__main__':
    main()
