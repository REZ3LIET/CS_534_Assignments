"""
CS 534 Assignment-2.2: K-Means Clustering Algorithm.

Author: Samar Kale
Date: 2025-10-22
"""

import random
import numpy as np
import matplotlib.pyplot as plt


class MBSAS:
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
    
    def get_sq_euclidean(self, ele_a: tuple, ele_b: tuple) -> float:
        """Calculates the squared euclidean distance of given elements"""
        x_ = (ele_a[0] - ele_b[0]) ** 2
        y_ = (ele_a[1] - ele_b[1]) ** 2
        return (x_ + y_)

    def update_representative(self, cluster: str, element: tuple) -> tuple:
        """Finds the new mean centroid of the given cluster"""
        n_c = len(self.clusters[cluster][0])
        m_c = self.clusters[cluster][-1]

        # Calculating new centroid axis-wise
        nm_x = ((n_c * m_c[0]) + element[0])/(n_c+1)
        nm_y = ((n_c * m_c[1]) + element[1])/(n_c+1)
        return (nm_x, nm_y)

    def find_cluster(self, element: tuple) -> str:
        """Finds the cluster centroid with closest to given element"""
        tmp_dist = np.inf
        cluster_name = ""

        # Loop to find closest cluster to element
        for cluster in self.clusters.keys():
            # Get centroid
            theta_j = self.clusters[cluster][-1]

            # Sq Euclidean distance between element and cluster centroid
            dist = self.get_sq_euclidean(element, theta_j)
            if dist < tmp_dist:
                tmp_dist = dist
                cluster_name = cluster

        return cluster_name

    def compute_mean(self, cluster_name: str) -> tuple:
        """Calculates the mean of a given clustered"""
        data_pts = self.clusters[cluster_name][0]
        n = len(data_pts)

        # If empty cluster return mean as infinity
        if n == 0:
            return (np.inf, np.inf)

        sum_x = 0
        sum_y = 0

        for x, y in data_pts:
            sum_x += x
            sum_y += y

        return (sum_x/n, sum_y/n)

    def has_converged(self, old, new, eps=1e-4):
        """Returns if the cluster's old and new centroids are within threshold"""
        return sum((a - b) ** 2 for a, b in zip(old, new)) < eps

    def run(self, data: tuple, m: int) -> dict:
        """
        Creates `m` new clusters for given data

        Args:
            data: input data to be analysed
            m: maximum number of clusters
        """
        if_converged = False  # Flag to check convergence
        itr = 0  # To record iterations

        # Initialize m arbitary clusters with unique centroids
        unique_data_pts = list(set(data))
        arbitary_centroids = random.sample(unique_data_pts, m)

        # Collection of clusters
        # Format: cluster id: (data list, representative)
        # For now the data list will be empty as only clusters
        # are to be created, w.r.t centroids
        for i in range(m):
            self.clusters[f"C_{i+1}"] = [[], arbitary_centroids[i]]

        for cnt in range(1000):
            # Reset assignments
            for cluster in self.clusters.values():
                cluster[0] = []

            # Phase - 1: Assign elements to cluster
            for i in range(len(data)):
                x_i = data[i]
                j_ = self.find_cluster(x_i)  # Get closest cluster to x_i
                self.clusters[j_][0].append(x_i)

            # Phase - 2: Update centroids of cluster w.r.t mean of elements in cluster
            converged_clusters = 0
            for j in self.clusters.keys():
                new_centroid = self.compute_mean(j)
                # if self.clusters[j][-1] == new_centroid:
                if self.has_converged(self.clusters[j][-1], new_centroid):
                    converged_clusters += 1
                    continue
                self.clusters[j][-1] = new_centroid

            # Convergence Criteria
            if converged_clusters == len(self.clusters):
                if_converged = True
                itr = cnt
                break

        if if_converged:
            print(f"K-Means Converged with {len(self.clusters)} clusters with {itr} iterations!")
        else:
            print("K-Means did not converge!")

        return self.clusters

    def plot_clusters(self, clusters: dict) -> None:
        """Plots the given clusters"""
        plt.figure(figsize=(8, 6))

        for label, (points, _) in clusters.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.scatter(xs, ys, label=label)  # Plot with label


        plt.title("K-Means Cluster")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    clusterer = MBSAS()
    data_path = "/workspaces/CS_534/CS_534_Assignments/Assignment_2/Data/cluster_data.txt"
    data = clusterer.load_data(data_path=data_path)
    clusters = clusterer.run(data, 20)
    clusterer.plot_clusters(clusters)
