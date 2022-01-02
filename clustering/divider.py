import os
import json
import glob
import itertools
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class Divider:

    def __init__(self, target_number):
        self.target_number = target_number
        self.labels = None

    def _find_optimal_k(self, data, nrefs=5, maxClusters=20):
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
        for gap_index, k in enumerate(range(1, maxClusters)):
            refDisps = np.zeros(nrefs)
            for i in range(nrefs):
                randomReference = np.random.random_sample(size=data.shape)
                km = KMeans(k)
                km.fit(randomReference)
                refDisp = km.inertia_
                refDisps[i] = refDisp
            km = KMeans(k)
            km.fit(data)

            origDisp = km.inertia_
            gap = np.log(np.mean(refDisps)) - np.log(origDisp)
            gaps[gap_index] = gap

            resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)
            result = (gaps.argmax() + 1, resultsdf)
        return result

    def show_gap_statistic(self, data, nrefs=5, maxClusters=25):
        score_g, df = self._find_optimal_k(reduced_data, nrefs, maxClusters)
        plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b')
        plt.xlabel('K')
        plt.ylabel('Gap Statistic')
        plt.title('Gap Statistic vs. K')

    def get_representatives(self, data):
        kmean = KMeans(n_clusters=self.target_number)
        kmean.fit(data)
        self.labels = kmean.labels_
        centers = np.array(kmean.cluster_centers_)

        print(Counter(labels))

        closest_data = []
        for i in range(self.target_number):
            center_vec = centers[i]
            data_idx_within_cluster = [idx for idx, clu_num in enumerate(m_clusters) if clu_num == i]

            one_cluster_tf_matrix = np.zeros((len(data_idx_within_cluster), centers.shape[1]))
            for row_num, data_idx in enumerate(data_idx_within_cluster):
                one_row = tf_matrix[data_idx]
                one_cluster_tf_matrix[row_num] = one_row

            closest, _ = pairwise_distances_argmin_min(center_vec, one_cluster_tf_matrix)
            closest_idx_in_one_cluster_tf_matrix = closest[0]
            closest_data_row_num = data_idx_within_cluster[closest_idx_in_one_cluster_tf_matrix]
            data_id = all_data[closest_data_row_num]

            closest_data.append(data_id)

        closest_data = list(set(closest_data))
        assert len(closest_data) == num_clusters

        return self.labels, closest_data

    def get_ISIC_atributes_summary(self, dataframe):
        summary = {}
        for value in range(0, self.target_number):
            rows = dataframe.loc[dataframe['cluster'] == value]
            summary[value] = {
              'milia_like_cyst': rows['milia_like_cyst'].sum(),
              'negative_network': rows['negative_network'].sum(),
              'streaks': rows['streaks'].sum(),
              'globules': rows['globules'].sum(),
              'pigment_network': rows['pigment_network'].sum()
            }
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.transpose()

        return summary_df

