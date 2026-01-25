package utils

func MeanNRecallAtK(pred [][]uint32, gt [][]uint32, n int, k int) float64 {
	var recalls []float64

	for i := 0; i < len(pred); i++ {
		predI := pred[i]
		gtI := gt[i]

		if len(predI) > n {
			predI = predI[:n]
		}
		if len(gtI) > k {
			gtI = gtI[:k]
		}

		predSet := make(map[uint32]bool)
		for _, val := range predI {
			predSet[val] = true
		}

		gtSet := make(map[uint32]bool)
		for _, val := range gtI {
			gtSet[val] = true
		}

		// Skip if gt is empty
		if len(gtSet) == 0 {
			continue
		}

		hits := 0
		for val := range gtSet {
			if predSet[val] {
				hits++
			}
		}

		recall := float64(hits) / float64(len(gtSet))
		recalls = append(recalls, recall)
	}

	if len(recalls) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, r := range recalls {
		sum += r
	}

	return sum / float64(len(recalls))
}
