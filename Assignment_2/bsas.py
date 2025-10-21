class BSAS:
    def __init__(self) -> None:
        pass

    def load_data(self, data_path: str) -> tuple:
        """Loads the data from .txt and converts to tuples"""
        read_data = ()
        return read_data
    
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