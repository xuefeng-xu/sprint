import heapq


class TopK:
    def __init__(self, k):
        self.k = k
        self.heap = []
        self.indices = set()

    def insert(self, dist, idx):
        # Avoid duplicates
        if idx in self.indices:
            return

        heapq.heappush(self.heap, (dist, idx))
        self.indices.add(idx)

        # Maintain only top-k elements
        if len(self.heap) > self.k:
            _, i = heapq.heappop(self.heap)
            self.indices.remove(i)

    def get_topk_indices(self):
        return [idx for _, idx in sorted(self.heap, reverse=True)]

    def get_topk_distances(self):
        return [dist for dist, _ in sorted(self.heap, reverse=True)]

    def get_topk_items(self):
        self.heap = sorted(self.heap, reverse=True)
        return self.heap
