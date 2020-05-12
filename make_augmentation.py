import tqdm
import numpy as np

from PIL import Image


def make_augmentation(landmark_file_name="/home/nasorokin11/Data/Data/data/train/landmarks.csv", train_path="/home"
                                                                                                            "/nasorokin11/Data/Data/data/train"
                                                                                                 "/images/"):
    """
    Adding gray scale and mirroring augmentation(
    :param landmark_file_name: path to landmarks
    :param train_path: path to train images
    :return: None
    """
    image_names = []
    new_landmarks = []
    print("start...")
    with open(landmark_file_name, "rt") as fp:
        for i, line in tqdm.tqdm(enumerate(fp)):
            if i == 0:
                continue  # skip header
            elements = line.strip().split("\t")
            image_name = elements[0]
            landmarks = list(map(np.int16, elements[1:]))
            landmarks = np.array(landmarks, dtype=np.int16).reshape((len(landmarks) // 2, 2))
            im = Image.open(train_path+"{}".format(image_name))
            # To grayscale
            im_gray = im.convert("LA").convert("RGB")
            im_gray.save(train_path+"gray_{}.jpg".format(i))
            image_names.append("gray_{}.jpg".format(i))

            new_landmarks.append(np.array(landmarks.reshape(landmarks.shape[0] * 2)))

            X = im.width
            im_mirror = Image.fromarray(np.array(im)[:, ::-1])
            im_mirror.save(train_path+"mirror_{}.jpg".format(i))
            image_names.append("mirror_{}.jpg".format(i))
            landmarks[:, 0] = X - landmarks[:, 0] - 1

            new_landmarks.append(np.array(landmarks.reshape(landmarks.shape[0] * 2)))

    print("save images...")
    with open(landmark_file_name, "a") as fl:
        for i in tqdm.tqdm(range(len(image_names))):
            fl.write(image_names[i] + "\t" + "\t".join(map(str, new_landmarks[i].tolist())) + "\n")


if __name__ == '__main__':
    make_augmentation()