"""
CS 534 Assignment-2.1: MBSAS Algorithm.

Author: Samar Kale
Date: 2025-10-21
"""

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

    def create_clusters(self, data: tuple, q: int, theta: float) -> None:
        """
        Creates new clusters for given data based on maximum clusters,
        threshold distance,

        Args:
            data: input data to be analysed
            q: maximum number of clusters
            theta: threshold distance beyond which new cluster is generated
        """
        m = 1  # Initial cluster

        # Collection of clusters
        # Format: cluster id: (data list, representative)
        # For now the data list will be empty as only clusters
        # are to be created, w.r.t representative
        self.clusters = {
            "C_1": [[], data[0]]
        }

        for i in range(2, len(data)+1):
            x_i = data[i-1]
            c_k = self.find_cluster(x_i)  # Get closest cluster to x_i

            # Distance between cluster rep and x_i
            dist = self.get_euclidean(x_i, self.clusters[c_k][-1])

            # Conditional check for new clusters
            if dist > theta and m < q:
                m += 1
                self.clusters[f"C_{m}"] = [[], x_i]
        
        print(f"{m} clusters created.")

    def assign_element_to_cluster(self, data: tuple) -> None:
        """
        Assigns the remaining elements to the created clusters

        Args:
            data: input data to be analysed
        """
        for i in range(len(data)):
            x_i = data[i]
            c_k = self.find_cluster(x_i)  # Get closest cluster to x_i

            # Update cluster with new element and representative
            self.clusters[c_k][0].append(x_i)
            self.clusters[c_k][-1] = self.update_representative(c_k, x_i)

    def run(self, data: tuple, q: int, theta: float) -> dict:
        """
        Finds the clusters for given data based on maximum clusters
        and threshold distance.

        Args:
            data: input data to be analysed
            q: maximum number of clusters
            theta: threshold distance eyond which new cluster is generated
        
        Returns:
            dict: dictionary of all the formulated clusters
        """
        self.create_clusters(data=data, q=q, theta=theta)
        self.assign_element_to_cluster(data=data)
        print("All elements have been assigned to the clusters")

        return self.clusters

    def plot_clusters(self, clusters: dict) -> None:
        """Plots the given clusters"""
        plt.figure(figsize=(8, 6))

        for label, (points, _) in clusters.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.scatter(xs, ys, label=label)  # Plot with label


        plt.title("MBSAS Cluster")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()

def estimate_clusters_with_mbsas(data_path, theta_range: int, q_max=1000, plot=True) -> None:
    """
    Estimate number of clusters m using MBSAS over a range of theta values.

    Args:
        data: data which is to be analysed
        theta_range: max number of theta value
        q_max: max number of clusters

    Returns:
        (theta_list, m_theta_list)
    """
    theta_list = np.arange(0.1, theta_range, 0.1).tolist()
    m_theta_list = []

    clusterer = MBSAS()
    data = clusterer.load_data(data_path=data_path)

    for i, theta in enumerate(theta_list):
        mid_cluster = []
        for _ in range(5):
            clusters = clusterer.run(data, q_max, theta)
            mid_cluster.append(len(clusters))

        avg_clusters = sum(mid_cluster)/len(mid_cluster)
        m_theta_list.append(avg_clusters)
        if avg_clusters == 1:
            print("Average clusters is now 1")
            break

    m_theta_vals = len(m_theta_list)
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(theta_list[:m_theta_vals], m_theta_list, marker='o', color="red")
        plt.xlabel('MBSAS threshold (Theta)')
        plt.ylabel('Number of clusters (m_theta)')
        plt.title('MBSAS Cluster Estimation')
        plt.grid(True)
        plt.show()

    print(f"Theta Values: {theta_list}")
    print(f"Cluster Values: {m_theta_list}")

if __name__ == "__main__":
    data_path = "/workspaces/CS_534/CS_534_Assignments/Assignment_2/Data/cluster_data.txt"
    estimate_clusters_with_mbsas(data_path, 10)
