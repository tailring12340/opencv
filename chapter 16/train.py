# python train.py --dataset data/digits.csv --model models/svm.cpickle

import joblib
from sklearn.svm import LinearSVC
from sklearn import svm
from pyimagesearch.hog import HOG
from pyimagesearch import dataset
import numpy as np
import argparse
import pickle


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "path to the dataset file")
ap.add_argument("-m", "--model", required = True,
	help = "path to where the model will be stored")
args = vars(ap.parse_args())

(digits, target) = dataset.load_digits(args["dataset"])
data = []

hog = HOG(orientations = 3, pixelsPerCell = (2, 2), cellsPerBlock = (4, 4), block_norm = 'L2-Hys')

for image in digits:
    image = dataset.deskew(image, 20)
    image = dataset.center_extent(image, (20, 20))

    hist = hog.describe(image)

    data.append(hist)
print("hist.shape", hist.shape)
data = np.array(data)
print("data.shape", data.shape)

model = LinearSVC(random_state = 42)
model.fit(data, target)

score = model.score(data, target)
print("score", score)


hist = np.reshape(hist, (1, -1))

test = model.predict(hist)[0]
print("result of test", test)
print("image", image)

joblib.dump(model, args["model"])