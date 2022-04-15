import joblib
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time

# This file is to train the model and also to get a precision score best on train test split #

# import local modules #
from definitions import features

# select what you need to run #
extract_features = "yes"
train_model = "yes"
accuracy = "no"
Kernel = "linear"

if extract_features == "yes":
    # extract features of images and save the features along with labels #
    feat, label = features("./myData", samples=4000)

    # save the features in a file #
    joblib.dump(feat, "model_features")
    joblib.dump(label, "model_label")
else:
    print("features already extracted")

# load the saved feature file #
feat = numpy.array(joblib.load("model_features"))
label = numpy.array(joblib.load("model_label")) # need to convert to 2d array for scaling #

# split into test and train dataset #
x_train, x_test, y_train, y_test = train_test_split(feat, label, test_size=0.1, shuffle=True)


# train the model #
if train_model == "yes":
    t0 = time()
    classifier = SVC(kernel=Kernel, gamma=0.00001, C=1)
    model = classifier.fit(x_train, y_train)
    t1 = time()
    print("Time taken to train the model: {} minutes".format((t1 - t0) / 60))

    # write in the time calc file #
    with open("time_calc.txt", "a") as f:
        f.writelines("\nTime taken to train the model with {} samples: {} minutes".format(len(x_train), (t1 - t0) / 60))
    joblib.dump(model, "trained_model")
else:
    print("Model is already trained")

# get accuracy scores #
if accuracy == "yes":
    t0 = time()
    init_data = "Samples used to train the model: {}\n"
    "Feature extractor: HOG\n"
    "Classifier Used: SVM\n"
    "Kernel: {}\n"
    "Samples used to test the model: {}\n".format(x_train.shape[0], Kernel, x_test.shape[0])
    trained_model = joblib.load("trained_model")
    pred = trained_model.predict(x_test)
    t1 = time()
    data = "Model accuracy score: {}% \nTime taken to Predict {} samples: {} minutes".format(
        round((accuracy_score(y_test, pred) * 100), 2), x_test.shape[0], round((t1 - t0) / 60))
    print(data)

    # write in the time calc file #
    with open("time_calc.txt", "a") as f:
        f.writelines("\n{}".format(init_data))
        f.writelines("\n{}".format(data))