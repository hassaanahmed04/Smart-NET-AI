import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def determine_optimal_clusters(coordinates, min_k=2, max_k=50):
    best_k = min_k
    best_score = -1
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(coordinates)
        score = silhouette_score(coordinates, kmeans.labels_)
        if score > best_score:
            best_k = k
            best_score = score
    return best_k

def optimize_tower_costs(cluster_centers, underserved_schools):
    costs = distance.cdist(underserved_schools, cluster_centers, metric='euclidean')
    n_schools, n_towers = costs.shape

    c = costs.flatten()
    A_eq_data, A_eq_rows, A_eq_cols = [], [], []

    for i in range(n_schools):
        for j in range(n_towers):
            A_eq_data.append(1)
            A_eq_rows.append(i)
            A_eq_cols.append(i * n_towers + j)

    A_eq = csr_matrix((A_eq_data, (A_eq_rows, A_eq_cols)), shape=(n_schools, n_schools * n_towers))
    b_eq = np.ones(n_schools)

    bounds = [(0, 1)] * (n_schools * n_towers)

    result = linprog(c, A_eq=A_eq.toarray(), b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        optimized_locations = result.x.reshape((n_schools, n_towers)).argmax(axis=1)
        return cluster_centers[optimized_locations]
    else:
        return cluster_centers

def cluster_and_optimize(schools, n_clusters):
    school_coordinates = schools[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(school_coordinates)
    schools['cluster'] = kmeans.labels_

    optimized_centers = []
    for cluster_id in range(n_clusters):
        cluster_schools = schools[schools['cluster'] == cluster_id][['latitude', 'longitude']].values
        cluster_centers = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        optimized = optimize_tower_costs(cluster_centers, cluster_schools)
        optimized_centers.extend(optimized)

    return np.array(optimized_centers)
