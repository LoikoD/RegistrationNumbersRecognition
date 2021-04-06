import tensorflow.keras as keras
import os
import random
import itertools
import numpy as np
import cv2
from imgaug import augmenters as iaa
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import OrderedEnqueuer


class DataGenerator(keras.utils.Sequence):
    def __init__(self, img_src_dir, samples=100, plates_list="plates_in_order.txt", chars_list="chars.txt",
                 img_shape=(256, 256), batch_size=4, shuffle=True, valid=False):
        self.images = []
        self.labels = []
        self.plates_list = open(plates_list).read().splitlines()
        self.chars_list = open(chars_list).read()
        self.src_dir = img_src_dir
        self.src_dir = os.path.join(self.src_dir, '')
        self.samples = samples
        self.img_shape = img_shape
        self.shuffle = shuffle
        self.valid = valid
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.images) / float(self.batch_size))

    def on_epoch_end(self):
        if not self.valid:
            self.images = []
            self.labels = []

            for folder, subs, files in os.walk(self.src_dir):
                check_files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
                if len(check_files) > 0:
                    counter = 0
                    if self.shuffle:
                        rnd = random.random() * 10000
                        random.Random(rnd).shuffle(check_files)
                    iter_files = itertools.cycle(check_files)

                    while counter < self.samples:
                        filename = next(iter_files)
                        entry = int(filename.split('.')[0]) - 1
                        license_plate = self.plates_list[entry]
                        label = [self.chars_list.index(symbol) for symbol in license_plate]
                        self.images.append(os.path.join(folder, filename))
                        self.labels.append(label)
                        counter += 1

            if self.shuffle:
                rnd = random.random() * 10000
                random.Random(rnd).shuffle(self.images)
                random.Random(rnd).shuffle(self.labels)

            self.images = np.array(self.images)
            self.labels = np.array(self.labels)
        else:
            self.images = []
            self.labels = []

            for folder, subs, files in os.walk(self.src_dir):
                check_files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
                if len(check_files) > 0:
                    counter = 0
                    if self.shuffle:
                        rnd = random.random() * 10000
                        random.Random(rnd).shuffle(check_files)
                    iter_files = itertools.cycle(check_files)

                    while counter < 5:
                        filename = next(iter_files)
                        entry = int(filename.split('.')[0]) - 1
                        license_plate = self.plates_list[entry]
                        label = [self.chars_list.index(symbol) for symbol in license_plate]
                        self.images.append(os.path.join(folder, filename))
                        self.labels.append(label)
                        counter += 1

            if self.shuffle:
                rnd = random.random() * 10000
                random.Random(rnd).shuffle(self.images)
                random.Random(rnd).shuffle(self.labels)

            self.images = np.array(self.images)
            self.labels = np.array(self.labels)

    def generate_data(self, indexes):

        images_batch = []
        labels_batch = []

        for i in indexes:
            image_name = os.path.join(self.src_dir, self.images[i])
            image = cv2.imread(image_name, cv2.IMREAD_COLOR)

            if self.valid:
                seq = iaa.Sequential([])
            else:
                seq = iaa.Sequential(
                    [
                        iaa.Crop(percent=(0, 0.1)),
                        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.5))),
                        iaa.LinearContrast((0.5, 1.5)),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.075 * 255), per_channel=0.5),
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.Sometimes(0.2, iaa.Rain(nb_iterations=(1, 1), speed=(0.04, 0.05), drop_size=(0.01, 0.01)))
                    ], random_order=True
                )
            seq_det = seq.to_deterministic()
            augmented_image = seq_det.augment_images([image])
            augmented_image = augmented_image[0]
            augmented_image = cv2.resize(augmented_image, self.img_shape)
            augmented_image = augmented_image / 255.
            images_batch.append(augmented_image)
            label = [to_categorical(num, len(self.chars_list)) for num in self.labels[i]]
            labels_batch.append(label)

            # ---- only for test ----
            # image = image / 255.
            # image = cv2.resize(image, self.img_shape)
            # images_batch.append(image)
            # labels_batch.append(label)
            # -----------------------

        images_batch = np.array(images_batch)
        labels_batch = np.array(labels_batch)
        return images_batch, labels_batch

    def __getitem__(self, item):
        indexes = [i + item * self.batch_size for i in range(self.batch_size)]
        a, la = self.generate_data(indexes)
        return a, la

    @staticmethod
    def label_to_text(label, chars_list="chars.txt"):
        chars_list = open(chars_list).read()
        plate_text = [chars_list[np.argmax(letter)] for letter in label]
        return "".join(plate_text)


if __name__ == "__main__":

    # change path, maybe for this:
    # src_dir = os.path.join(os.path.dirname(__file__), r'images')
    src_dir = r'F:\samples\images'
    # ----------------------------------

    train_gen = DataGenerator(img_src_dir=src_dir, samples=100, img_shape=(512, 512), batch_size=4)
    enqueuer = OrderedEnqueuer(train_gen)
    enqueuer.start(workers=1, max_queue_size=4)
    output_gen = enqueuer.get()

    gen_len = len(train_gen)
    try:
        for b in range(gen_len):
            batch = next(output_gen)
            for img, lbl in zip(batch[0], batch[1]):
                print(lbl)
                print(DataGenerator.label_to_text(lbl))
                cv2.imshow("win", img)
                cv2.waitKey(0)
    finally:
        enqueuer.stop()
