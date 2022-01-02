import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def get_features(filename):
    features = []
    names = []

    f = pd.read_json(filename)
    raw_features = f['feature_vector']
    raw_names = f['fname']

    for name in raw_names:
        name = name[-19:]
        names.append(name)

    for feature in raw_features:
        feature = np.array(feature)
        features.append(feature)

    features = np.reshape(features, (len(features), -1))
    return features, names


def preprocess(data):
    min_max = MinMaxScaler()
    data = min_max.fit_transform(data)

    pca = PCA().fit(data)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.99) + 1
    print("{0} components were left".format(d))
    pca = PCA(n_components=d)
    reduced_data = pca.fit_transform(data)
    return reduced_data


def get_attributes_df(cropped_names, attributes_dir, attributes_names):
    attributes_files = [os.path.basename(x) for x in glob.glob(attributes_dir + '**/*.png')]

    attributes_dict = {}
    for n in cropped_names:
        attributes_dict[n] = {}
        for key in attributes_names:
            attributes_dict[n][key] = None

    index = [idx for idx, s in enumerate(attributes_files) if cropped_names[0] in s]

    for name in cropped_names:
        list_of_attributes_files = [attributes_files[idx] for idx, s in enumerate(attributes_files) if name in s]
        for a in list_of_attributes_files:
            im = np.array(Image.open(os.path.join(attributes_dir, a)))
            idx = np.where(list(map(a.__contains__, attributes_names)))[0][0]
            if 255 in im:
                attributes_dict[name][attributes_names[idx]] = True
            else:
                attributes_dict[name][attributes_names[idx]] = False

    attributes_df = pd.DataFrame(attributes_dict)
    attributes_df = attributes_df.transpose()
    return attributes_df
