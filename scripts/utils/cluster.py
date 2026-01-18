import numpy as np

def cluster_point_cloud(pcd, eps=0.02, min_points=10, render_boxes=False):
    """
    Cluster the point cloud using DBSCAN clustering.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        eps (float): The maximum distance between two points to be considered in the same neighborhood.
        min_points (int): The minimum number of points required to form a cluster.

    Returns:
        List[o3d.geometry.PointCloud]: A list of clustered point clouds.
    """
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    max_label = labels.max()
    clusters = []
    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        cluster = pcd.select_by_index(indices)
        clusters.append(cluster)
        
    cluster_boxes = []
    if render_boxes:
        for cluster in clusters:
            bbox = cluster.get_oriented_bounding_box()
            bbox.color = (0.2, 0.8, 0.2)
            cluster_boxes.append(bbox)

    return cluster_boxes, clusters