import os
import sys
import argparse
from keras.models import Model
import keras
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
import tensorflow as tf

K.clear_session()

def main(argv):
    parser = argparse.ArgumentParser()

    # Optional arguments.
    parser.add_argument(
        "--size",
        default=224,
        help="The image size of train sample.")
    parser.add_argument(
        "--batch",
        default=32,
        help="The number of train samples per batch.")
    parser.add_argument(
        "--epochs",
        default=300,
        help="The number of train iterations.")
    parser.add_argument(
        "--path",
        default="/content/gdrive/My Drive/Colab Notebooks/data/",
        help="The path to dataset.")
    parser.add_argument(
        "--weights",
        default="",
        help="The path to weights.")

    args = parser.parse_args()

    train(int(args.batch), int(args.epochs), get_classses(str(args.path)),int(args.size), str(args.path), str(args.weights))


def get_classses(path_data):
    labels = os.listdir(os.path.join(path_data, "train"))
    with open("label.txt", "w") as f:
        for i in labels:
            f.write(str(i + '\n'))
    return len(labels)


def MobileNet_V2( k):
    mobile = keras.applications.mobilenet_v2.MobileNetV2(alpha=1.4)
    x = mobile.layers[-2].output
    preditcions = Dense(k, activation="softmax")(x)
    model = Model(inputs=mobile.input, outputs=preditcions)
    for layer in model.layers[:-25]:
        layer.trainable = False
    # model.summary()
    return model


def generate(batch, size, path_data):
    ptrain = os.path.join(path_data, "train")
    pval = os.path.join(path_data, "val")
    test = os.path.join(path_data, "test")
    print(ptrain)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        ptrain,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        pval,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    _, _, train_files = os.walk(ptrain)
    _, _, val_files = os.walk(pval)
    _, _, test_files = os.walk(test)

    return train_generator, validation_generator, test_generator, len(train_files), len(val_files), len(test_files)


def train(batch, epochs, num_classes, size, path_data, weights):
    train_generator, validation_generator, test_generator, count1, count2, count3 = generate(batch, size, path_data)
    model = MobileNet_V2 (num_classes)
    log_dir = '/content/gdrive/My Drive/Colab Notebooks/logs/000/'
    opt = Adam(lr=1e-5)
    earlystop = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3, verbose=1,
                                 mode='max')
    if weights != "":
        model.load_weights(weights)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=count1 // batch,
        validation_steps=count2 // batch,
        epochs=epochs,
        callbacks=[earlystop, logging, checkpoint, reduce_lr])

    if not os.path.exists('model'):
        os.makedirs('model')
    scores = model.evaluate_generator(test_generator, count3 // batch)
    print("Аккуратность на тестовых данных: %.2f%%" % (scores[1] * 100))
    # model.save_weights('/content/gdrive/My Drive/Colab Notebooks/MobileNetV2-master/model/weights.h5')
    model.save("/content/gdrive/My Drive/Colab Notebooks/MobileNetV2-master/model/model_MIRO_M2.h5")


if __name__ == '__main__':
    main(sys.argv)
