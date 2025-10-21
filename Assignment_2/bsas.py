class BSAS:
    def __init__(self) -> None:
        pass

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
    
    def find_cluster(self, element) -> str:
        """Finds the cluster representative with closest to given element"""
        cluster_name = ""
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
        clusters = {}
        return clusters
    
    def plot_clusters(self, clusters: dict) -> None:
        """Plots the given clusters"""
        pass

if __name__ == "__main__":
    clusterer = BSAS()
    data_path = "/workspaces/CS_534/CS_534_Assignments/Assignment_2/Data/cluster_data.txt"
    data = clusterer.load_data(data_path=data_path)
    print(data[-5:])
