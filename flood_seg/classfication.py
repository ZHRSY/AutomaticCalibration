from sklearn.cluster import KMeans
import numpy as np
import pandas as pd



class RunoffClassifier:
    #聚类特征：最大降雨、最大流量
    def __init__(self, config, column_num):
        self.n_clusters = config.get('n_clusters', 3)
        self.column_num = column_num
        self.result = []
        self.path_out = config.get('path_out', '')
        self.rain_file_paths = config.get('rain_file_paths', [])

    def classify(self, result):
        df = pd.read_csv(self.path_out, encoding="utf-8-sig", encoding_errors='ignore')
        targets = df.iloc[:, self.column_num + 1].values
        rainfall_df = pd.read_csv(self.rain_file_paths, encoding="utf-8-sig", encoding_errors='ignore')
        rainfall = rainfall_df.iloc[:, self.column_num].values
        features = []
        for interval in result:
            rain_segment = rainfall[interval[0]:interval[1]]
            max_rain = np.max(rain_segment)
            max_runoff = np.max(targets[interval[0]:interval[1]])
            features.append([max_rain, max_runoff])

        features_array = np.array(features)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(features_array)
        labels = kmeans.labels_

        clustered_results = {i: [] for i in range(self.n_clusters)}
        clustered_features = {i: [] for i in range(self.n_clusters)}
        for idx, label in enumerate(labels):
            clustered_results[label].append(result[idx])
            clustered_features[label].append(features[idx])

        cluster_max_rainfall = {i: np.mean(clustered_features[i]) for i in range(len(clustered_features))}
        sorted_clusters = sorted(cluster_max_rainfall, key=cluster_max_rainfall.get)

        results = [clustered_results[cluster] for cluster in sorted_clusters]
        clustered_features = [clustered_features[cluster] for cluster in sorted_clusters]

        return results, clustered_features
    
    def char(self, features):
        print(f'洪水场次聚类共分为{self.n_clusters}类')
        for i in range (self.n_clusters):
            first_0 = [item[0] for item in features[i]]
            first_1 =  [item[1] for item in features[i]]
            
            print(f"特征{i+1}: 共{len(features[i])}场洪水，平均最大降雨{np.mean(first_0)}，平均最大流量{np.mean(first_1)}",)



if __name__ == '__main__':
    config = {
        "n_clusters" : 4,
    "flow_file_paths":'./data/new.csv',  #实际流量数据
    "rain_file_paths": "./data/basin_rainfall.csv",  #降雨数据
    "path_out": './data/post_proce.csv', #处理后流量数据
    "column_num": 1,  #流域第几列
    "theta": 24  #默认
}
    classifier = RunoffClassifier(config)
    flood_segmentation = FloodSegmentation(config)
    result = flood_segmentation.process()
    results, clustered_features = classifier.classify(result)
    # print(results)
    print(clustered_features)
    print(np.shape(clustered_features[0]))
    print(np.shape(clustered_features[1]))
    print(np.shape(clustered_features[2]))
    print(np.shape(clustered_features[3]))
    
    