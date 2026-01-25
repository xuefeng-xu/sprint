package utils

import (
	"container/heap"
	"sort"
)

type Item struct {
	Dist  float32
	Index uint32
}

type MinHeap []Item

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].Dist < h[j].Dist }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MinHeap) Push(x interface{}) {
	*h = append(*h, x.(Item))
}

func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

type TopK struct {
	k       int
	heap    *MinHeap
	indices map[uint32]bool
}

func NewTopK(k int) *TopK {
	h := &MinHeap{}
	heap.Init(h)
	return &TopK{
		k:       k,
		heap:    h,
		indices: make(map[uint32]bool),
	}
}

func (t *TopK) Insert(dist float32, idx uint32) {
	if t.indices[idx] {
		return
	}

	heap.Push(t.heap, Item{Dist: dist, Index: idx})
	t.indices[idx] = true

	if t.heap.Len() > t.k {
		item := heap.Pop(t.heap).(Item)
		delete(t.indices, item.Index)
	}
}

func (t *TopK) GetTopKIndices() []uint32 {
	items := make([]Item, len(*t.heap))
	copy(items, *t.heap)

	sort.Slice(items, func(i, j int) bool {
		return items[i].Dist > items[j].Dist
	})

	indices := make([]uint32, len(items))
	for i, item := range items {
		indices[i] = item.Index
	}
	return indices
}

func (t *TopK) GetTopKDistances() []float32 {
	items := make([]Item, len(*t.heap))
	copy(items, *t.heap)

	sort.Slice(items, func(i, j int) bool {
		return items[i].Dist > items[j].Dist
	})

	distances := make([]float32, len(items))
	for i, item := range items {
		distances[i] = item.Dist
	}
	return distances
}

func (t *TopK) GetTopKItems() []Item {
	items := make([]Item, len(*t.heap))
	copy(items, *t.heap)

	sort.Slice(items, func(i, j int) bool {
		return items[i].Dist > items[j].Dist
	})

	return items
}
