"""
CS 534 Assignment-2: BSAS Algorithm.

Author: Samar Kale
Date: 2025-10-21
"""

import numpy as np
import matplotlib.pyplot as plt


class BSAS:
    def __init__(self) -> None:
        self.clusters = {}

    def load_data(self, data_path: str) -> tuple:
        """Loads the data from given text file and converts to tuples"""
        data = []

        # Read the data and store it into a list
        with open(data_path, "r") as f:
            raw_data = f.readlines()

        # Convert the data into float and remove the first column (Serial No.)
        for line in raw_data:
            _, x_, y_ = line.split()
            data.append((float(x_), float(y_)))

        # Return data as tuple
        return tuple(data)
    
    def get_euclidean(self, ele_a: tuple, ele_b: tuple) -> float:
        """Calculates the euclidean distance of given elements"""
        x_ = (ele_a[0] - ele_b[0]) ** 2
        y_ = (ele_a[1] - ele_b[1]) ** 2
        return (x_ + y_) ** 0.5

    def update_representative(self, cluster: str, element: tuple) -> tuple:
        """Finds the new mean centroid of the given cluster"""
        n_c = len(self.clusters[cluster][0])
        m_c = self.clusters[cluster][-1]

        # Calculating new centroid axis-wise
        nm_x = ((n_c * m_c[0]) + element[0])/(n_c+1)
        nm_y = ((n_c * m_c[1]) + element[1])/(n_c+1)
        return (nm_x, nm_y)

    def find_cluster(self, element: tuple) -> str:
        """Finds the cluster representative with closest to given element"""
        tmp_dist = np.inf
        cluster_name = ""

        # Loop to find closest cluster to element
        for cluster in self.clusters.keys():
            # Get representative
            c_j = self.clusters[cluster][-1]

            # Euclidean distance between element and cluster representative
            dist = self.get_euclidean(element, c_j)
            if dist < tmp_dist:
                tmp_dist = dist
                cluster_name = cluster

        return cluster_name
    
    def bsas_algorithm(self, data: tuple, q: int, theta: float) -> dict:
        """
        Finds the clusters for given data based on maximum clusters,
        threshold distance,

        Args:
            q: maximum number of clusters
            theta: threshold distance eyond which new cluster is generated
        
        Returns:
            dict: dictionary of all the formulated clusters
        """
        m = 1  # Initial cluster
        self.clusters = {  # Collection of clusters
            "C_1": [[data[0]], data[0]]  # cluster id: (data list, representative)
        }

        for i in range(2, len(data)+1):
            x_i = data[i-1]
            c_k = self.find_cluster(x_i)  # Get closest cluster to x_i

            # Distance between cluster rep and x_i
            dist = self.get_euclidean(x_i, self.clusters[c_k][-1])

            # Validity check
            if dist > theta and m < q:
                m += 1
                self.clusters[f"C_{m}"] = [[x_i], x_i]
            else:
                self.clusters[c_k][0].append(x_i)
                self.clusters[c_k][-1] = self.update_representative(c_k, x_i)

        return self.clusters
    
    def plot_clusters(self, clusters: dict) -> None:
        """Plots the given clusters"""
        plt.figure(figsize=(8, 6))

        for label, (points, _) in clusters.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.scatter(xs, ys, label=label)  # Plot with label


        plt.title("BSAS Cluster")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    clusterer = BSAS()
    data_path = "/workspaces/CS_534/CS_534_Assignments/Assignment_2/Data/cluster_data.txt"
    data = clusterer.load_data(data_path=data_path)
    clusters = clusterer.bsas_algorithm(data, 20, 0.5)
    clusterer.plot_clusters(clusters)
