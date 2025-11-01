import math
import numpy as np
from k_means_constrained import KMeansConstrained


def balanced_kmeans(x, n_clusters, random_state=0):
    n, _ = x.shape

    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=n // n_clusters,
        size_max=math.ceil(n / n_clusters),
        random_state=random_state,
        n_jobs=-1,
    )
    kmeans.fit(x)

    bincount = np.bincount(kmeans.labels_)
    maxcount = np.max(bincount)

    parents = []
    children = []
    for i in range(n_clusters):
        mask = kmeans.labels_ == i

        # Select the closest vector as the center
        dist_i = x[mask].dot(kmeans.cluster_centers_[i])
        parent_i = np.argmax(dist_i)

        # Map back to original indices
        parent_i = np.flatnonzero(mask)[parent_i]
        parents.append(parent_i)

        if bincount[i] < maxcount:
            diff = int(maxcount - bincount[i])

            # Find the closest vectors from other clusters
            dist_other_i = x[~mask].dot(x[parent_i])
            supp_idx = np.argpartition(dist_other_i, -diff)[-diff:]

            # Add them to this cluster to make it balanced
            supp_idx = np.flatnonzero(~mask)[supp_idx]
            mask[supp_idx] = True

        # Store children indices
        mask[parent_i] = False
        children_i = np.flatnonzero(mask)
        children.append(children_i)

    parents = np.asarray(parents)

    return parents, children


def hier_balanced_kmeans(x, n0, branch, random_state=0):
    # Top level
    indices_0, children = balanced_kmeans(x, n0, random_state=random_state)
    indices = [indices_0]

    # Top-down construction
    while len(children[0]) >= 3 * branch:
        indices_i = []
        children_i = []

        for child_j in children:
            indices_j, children_j = balanced_kmeans(
                x[child_j], branch, random_state=random_state
            )

            # Map back to original indices
            indices_i.extend(child_j[indices_j])

            # Store children indices for next level
            for c in children_j:
                children_i.append(child_j[c])

        children = children_i
        indices.append(indices_i)

    # Bottom level
    children = np.asarray(children).flatten()
    if children.size > 0:
        indices.append(children)

    return indices
