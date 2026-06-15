package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"time"

	"pianopir"
	"sprint/go/utils"
)

type BatchResult struct {
	N0        int     `json:"n0"`
	Branch    int     `json:"branch"`
	BeamWidth int     `json:"beamw"`
	Cached    int     `json:"cached"`
	BatchSize int     `json:"batchsize"`
	Recall10  float64 `json:"recall@10"`
	Time      float64 `json:"time"`
}

func batchPaddedEntryByteCount(dim int) uint64 {
	base := uint64(dim) * 4
	return (base + 31) / 32 * 32
}

func batchLevelBatchSize(levelSize int, branch int, beamWidth int) uint64 {
	if levelSize <= 0 {
		return 2
	}
	if branch <= 0 {
		branch = 1
	}
	if beamWidth <= 0 {
		beamWidth = 1
	}

	want := beamWidth * branch
	if want < 1 {
		want = 1
	}
	if want > levelSize {
		want = levelSize
	}

	batchSize := uint64(want)
	if batchSize < 2 {
		batchSize = 2
	}
	if batchSize%2 != 0 {
		batchSize++
	}

	maxSafe := uint64(levelSize) * 2
	if batchSize > maxSafe {
		batchSize = maxSafe
		if batchSize < 2 {
			batchSize = 2
		}
		if batchSize%2 != 0 {
			batchSize--
			if batchSize < 2 {
				batchSize = 2
			}
		}
	}

	return batchSize
}

func batchPackVectorToWords(vec []float32, entrySizeWords uint64) []uint64 {
	entryBytes := make([]byte, entrySizeWords*8)
	for j := 0; j < len(vec); j++ {
		binary.LittleEndian.PutUint32(entryBytes[j*4:], math.Float32bits(vec[j]))
	}
	words := make([]uint64, entrySizeWords)
	for w := uint64(0); w < entrySizeWords; w++ {
		words[w] = binary.LittleEndian.Uint64(entryBytes[w*8:])
	}
	return words
}

func batchDotPacked(words []uint64, xq []float32) float32 {
	sum := float32(0)
	for j := 0; j < len(xq); j++ {
		word := words[j/2]
		var v float32
		if j%2 == 0 {
			v = math.Float32frombits(uint32(word & 0xffffffff))
		} else {
			v = math.Float32frombits(uint32((word >> 32) & 0xffffffff))
		}
		sum += xq[j] * v
	}
	return sum
}

func batchBuildLevelPIRs(xb [][]float32, indices [][]uint32, branch int, beamWidth int, cached int) ([]*pianopir.SimpleBatchPianoPIR, uint64) {
	if len(xb) == 0 {
		log.Fatalf("empty base vectors")
	}

	dim := len(xb[0])
	entryByteCount := batchPaddedEntryByteCount(dim)
	entrySizeWords := entryByteCount / 8

	levelPIRs := make([]*pianopir.SimpleBatchPianoPIR, len(indices))
	for lvl := 0; lvl < len(indices); lvl++ {
		if lvl < cached {
			continue
		}

		lvlCount := len(indices[lvl])
		if lvlCount == 0 {
			continue
		}

		rawDB := make([]uint64, uint64(lvlCount)*entrySizeWords)
		for pos := 0; pos < lvlCount; pos++ {
			globalIdx := indices[lvl][pos]
			vec := xb[globalIdx]
			packed := batchPackVectorToWords(vec, entrySizeWords)
			copy(rawDB[uint64(pos)*entrySizeWords:uint64(pos+1)*entrySizeWords], packed)
		}

		batchSize := batchLevelBatchSize(lvlCount, branch, beamWidth)
		pir := pianopir.NewSimpleBatchPianoPIR(uint64(lvlCount), entryByteCount, batchSize, rawDB, 8)
		pir.Preprocessing()
		levelPIRs[lvl] = pir
	}

	return levelPIRs, entrySizeWords
}

func batchQueryAll(pir *pianopir.SimpleBatchPianoPIR, batch []uint64) ([][]uint64, time.Duration, error) {
	if len(batch) == 0 {
		return [][]uint64{}, 0, nil
	}

	cfg := pir.Config()
	if cfg.BatchSize == 0 {
		return nil, 0, fmt.Errorf("invalid BatchPIR config (BatchSize=0): %+v", cfg)
	}
	if cfg.DBSize == 0 {
		return nil, 0, fmt.Errorf("invalid BatchPIR config (DBSize=0): %+v", cfg)
	}
	if cfg.PartitionNum == 0 {
		return nil, 0, fmt.Errorf("invalid BatchPIR config (PartitionNum=0): %+v", cfg)
	}
	if cfg.PartitionSize == 0 {
		return nil, 0, fmt.Errorf("invalid BatchPIR config (PartitionSize=0): %+v", cfg)
	}
	if cfg.BatchSize < cfg.PartitionNum {
		return nil, 0, fmt.Errorf("invalid BatchPIR config (BatchSize < PartitionNum): %+v", cfg)
	}

	servedPerPartition := int(cfg.BatchSize / cfg.PartitionNum)
	if servedPerPartition <= 0 {
		return nil, 0, fmt.Errorf("invalid BatchPIR derived servedPerPartition=%d: %+v", servedPerPartition, cfg)
	}

	positionsByIdx := make(map[uint64][]int, len(batch))
	for pos, idx := range batch {
		if idx >= cfg.DBSize {
			return nil, 0, fmt.Errorf("batch index out of range: idx=%d DBSize=%d", idx, cfg.DBSize)
		}
		positionsByIdx[idx] = append(positionsByIdx[idx], pos)
	}

	pendingByPartition := make([][]uint64, cfg.PartitionNum)
	pendingAdded := make(map[uint64]bool, len(positionsByIdx))
	for _, idx := range batch {
		if pendingAdded[idx] {
			continue
		}
		pendingAdded[idx] = true
		partitionIdx := idx / cfg.PartitionSize
		if partitionIdx >= cfg.PartitionNum {
			return nil, 0, fmt.Errorf("computed partition out of range: idx=%d partitionIdx=%d PartitionNum=%d PartitionSize=%d", idx, partitionIdx, cfg.PartitionNum, cfg.PartitionSize)
		}
		pendingByPartition[partitionIdx] = append(pendingByPartition[partitionIdx], idx)
	}

	out := make([][]uint64, len(batch))
	done := make(map[uint64]bool, len(positionsByIdx))
	pendingCount := len(positionsByIdx)
	queryTime := time.Duration(0)

	for pendingCount > 0 {
		round := make([]uint64, 0, cfg.BatchSize)

		for part := uint64(0); part < cfg.PartitionNum; part++ {
			filler := part * cfg.PartitionSize
			if filler >= cfg.DBSize {
				filler = cfg.DBSize - 1
			}

			picked := 0
			for picked < servedPerPartition && len(pendingByPartition[part]) > 0 {
				idx := pendingByPartition[part][0]
				pendingByPartition[part] = pendingByPartition[part][1:]
				if done[idx] {
					continue
				}
				round = append(round, idx)
				picked++
			}
			for picked < servedPerPartition {
				round = append(round, filler)
				picked++
			}
		}

		if uint64(len(round)) < cfg.BatchSize {
			extra := int(cfg.BatchSize) - len(round)
			for i := 0; i < extra; i++ {
				round = append(round, 0)
			}
		}

		if uint64(len(round)) != cfg.BatchSize {
			return nil, queryTime, fmt.Errorf("internal error: constructed round of size %d want %d", len(round), cfg.BatchSize)
		}

		var responses [][]uint64
		var err error
		for attempt := 0; attempt < 5; attempt++ {
			onlineStart := time.Now()
			responses, err = pir.Query(round)
			onlineElapsed := time.Since(onlineStart)
			if err == nil {
				queryTime += onlineElapsed
				break
			}
			pir.Preprocessing()
		}
		if err != nil {
			return nil, queryTime, err
		}
		if len(responses) != len(round) {
			return nil, queryTime, fmt.Errorf("BatchPIR.Query response length mismatch: got %d want %d", len(responses), len(round))
		}

		for i, idx := range round {
			posList, ok := positionsByIdx[idx]
			if !ok || done[idx] {
				continue
			}
			done[idx] = true
			pendingCount--
			for _, pos := range posList {
				out[pos] = responses[i]
			}
		}

		if pir.FinishedBatchNum+10 >= pir.SupportBatchNum {
			pir.Preprocessing()
		}
	}

	return out, queryTime, nil
}

func batchSearchQueries(xq [][]float32, xb [][]float32, indices [][]uint32, beamWidth int, cached int, k int, levelPIRs []*pianopir.SimpleBatchPianoPIR, entrySizeWords uint64) ([][]uint32, time.Duration) {
	batchSize := len(xq)
	if batchSize == 0 {
		return [][]uint32{}, 0
	}

	topkCand := make([]*utils.TopK, batchSize)
	for i := 0; i < batchSize; i++ {
		topkCand[i] = utils.NewTopK(k)
	}

	candPos := make([]uint32, 0)
	serverTime := time.Duration(0)

	for lvl, indicesLvl := range indices {
		distRows := make([][]float32, batchSize)
		for j := 0; j < batchSize; j++ {
			distRows[j] = make([]float32, len(indicesLvl))
		}

		if lvl < cached {
			for col, globalIdx := range indicesLvl {
				vec := xb[globalIdx]
				for q := 0; q < batchSize; q++ {
					var dist float32
					for d := 0; d < len(xq[q]); d++ {
						dist += xq[q][d] * vec[d]
					}
					distRows[q][col] = dist
				}
			}
		} else {
			if len(candPos) == 0 {
				log.Fatalf("non-cached level %d but candPos is empty", lvl)
			}
			if lvl == 0 {
				log.Fatalf("non-cached level cannot be 0")
			}

			branchFactor := len(indicesLvl) / len(indices[lvl-1])
			if branchFactor <= 0 {
				log.Fatalf("invalid branchFactor at level %d", lvl)
			}

			localIdx := make([]uint32, 0, len(candPos)*branchFactor)
			for _, p := range candPos {
				start := int(p) * branchFactor
				for off := 0; off < branchFactor; off++ {
					localIdx = append(localIdx, uint32(start+off))
				}
			}

			pir := (*pianopir.SimpleBatchPianoPIR)(nil)
			if lvl < len(levelPIRs) {
				pir = levelPIRs[lvl]
			}
			if pir == nil {
				log.Fatalf("missing PIR instance for level %d (cached=%d)", lvl, cached)
			}

			batch := make([]uint64, len(localIdx))
			for i := range localIdx {
				batch[i] = uint64(localIdx[i])
			}

			responses, onlineTime, err := batchQueryAll(pir, batch)
			if err != nil {
				log.Fatalf("BatchPIR.Query(level=%d, n=%d) failed: %v", lvl, len(batch), err)
			}
			if len(responses) != len(batch) {
				log.Fatalf("BatchPIR.Query(level=%d) response length mismatch: got %d want %d", lvl, len(responses), len(batch))
			}
			serverTime += onlineTime

			for col := range responses {
				resp := responses[col]
				if uint64(len(resp)) < entrySizeWords {
					log.Fatalf("BatchPIR response too small: got %d words want %d", len(resp), entrySizeWords)
				}
				for q := 0; q < batchSize; q++ {
					distRows[q][col] = batchDotPacked(resp[:entrySizeWords], xq[q])
				}
			}

			// Build per-query sorted candidate order
			idxMatrix := make([][]uint32, batchSize)
			for q := 0; q < batchSize; q++ {
				order := make([]uint32, len(localIdx))
				for i := range order {
					order[i] = uint32(i)
				}
				sort.Slice(order, func(i, j int) bool {
					return distRows[q][order[i]] > distRows[q][order[j]]
				})
				idxMatrix[q] = order

				// populate per-query TopK (global indices)
				for _, colIdx := range idxMatrix[q] {
					globalIdx := indicesLvl[localIdx[colIdx]]
					topkCand[q].Insert(distRows[q][colIdx], globalIdx)
				}
			}

			// Aggregate top candidates across the batch into a shared TopK
			// For each candidate position, find max distance across all queries
			if lvl < len(indices)-1 {
				topkAll := utils.NewTopK(beamWidth)
				for col := 0; col < len(localIdx); col++ {
					maxDist := float32(math.Inf(-1))
					for q := 0; q < batchSize; q++ {
						if distRows[q][col] > maxDist {
							maxDist = distRows[q][col]
						}
					}
					topkAll.Insert(maxDist, localIdx[col])
				}
				candPos = topkAll.GetTopKIndices()
			}
		}

		if lvl < cached {
			// Populate per-query TopK for cached level
			for q := 0; q < batchSize; q++ {
				order := make([]uint32, len(indicesLvl))
				for i := range order {
					order[i] = uint32(i)
				}
				sort.Slice(order, func(i, j int) bool {
					return distRows[q][order[i]] > distRows[q][order[j]]
				})

				for _, idx := range order {
					topkCand[q].Insert(distRows[q][idx], indicesLvl[idx])
				}
			}

			// Aggregate batch-level TopK using max distance across queries per candidate
			if lvl < len(indices)-1 {
				topkAll := utils.NewTopK(beamWidth)
				for col := uint32(0); col < uint32(len(indicesLvl)); col++ {
					maxDist := float32(math.Inf(-1))
					for q := 0; q < batchSize; q++ {
						if distRows[q][col] > maxDist {
							maxDist = distRows[q][col]
						}
					}
					topkAll.Insert(maxDist, col)
				}
				candPos = topkAll.GetTopKIndices()
			}
		}

		if lvl < len(indices)-1 && beamWidth > 0 && len(candPos) > beamWidth {
			candPos = candPos[:beamWidth]
		}
	}

	results := make([][]uint32, batchSize)
	for i := 0; i < batchSize; i++ {
		results[i] = topkCand[i].GetTopKIndices()
	}

	return results, serverTime
}

func batchGetProjectRoot() string {
	_, filename, _, _ := runtime.Caller(0)
	return filepath.Dir(filepath.Dir(filepath.Dir(filename)))
}

func batchSaveResults(dataset string, n0 int, branch int, beamWidth int, cached int, batchSize int, recall10 float64, serverTime float64) error {
	projectRoot := batchGetProjectRoot()
	resultDir := filepath.Join(projectRoot, "result", dataset)

	if err := os.MkdirAll(resultDir, 0755); err != nil {
		return err
	}

	result := BatchResult{
		N0:        n0,
		Branch:    branch,
		BeamWidth: beamWidth,
		Cached:    cached,
		BatchSize: batchSize,
		Recall10:  recall10,
		Time:      serverTime,
	}

	jsonBytes, err := json.Marshal(result)
	if err != nil {
		return err
	}

	filePath := filepath.Join(resultDir, "batch.jsonl")
	f, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.Write(append(jsonBytes, '\n'))
	return err
}

func main() {
	dataset := flag.String("dataset", "movielens", "Dataset name")
	n0 := flag.Int("n0", 100, "N0 parameter")
	branch := flag.Int("branch", 2, "Branch parameter")
	beamWidth := flag.Int("beamw", 1, "Beam width for batch search")
	cached := flag.Int("cached", 1, "Cached index depth")
	batchSize := flag.Int("batchsize", 10, "Batch size for search")
	flag.Parse()

	xb := utils.LoadBase(*dataset)
	xq := utils.LoadQuery(*dataset)
	gt := utils.LoadGroundtruth(*dataset)
	indices := utils.LoadIndices(*dataset, *n0, *branch)

	levelPIRs, entrySizeWords := batchBuildLevelPIRs(xb, indices, *branch, *beamWidth, *cached)

	if len(xq) == 0 {
		log.Fatalf("no queries loaded for dataset %s", *dataset)
	}

	bsize := *batchSize
	if bsize <= 0 {
		bsize = len(xq)
	}

	numBatches := int(math.Ceil(float64(len(xq)) / float64(bsize)))
	totalServerTime := time.Duration(0)
	finalResults := make([][]uint32, 0, len(xq))

	for b := 0; b < numBatches; b++ {
		start := b * bsize
		end := (b + 1) * bsize
		if end > len(xq) {
			end = len(xq)
		}
		xqBatch := xq[start:end]

		resultsBatch, serverTime := batchSearchQueries(xqBatch, xb, indices, *beamWidth, *cached, 10, levelPIRs, entrySizeWords)
		totalServerTime += serverTime
		finalResults = append(finalResults, resultsBatch...)
	}

	recall10 := utils.MeanNRecallAtK(finalResults, gt[:len(finalResults)], 10, 10)
	avgBatchTime := totalServerTime.Seconds() / float64(numBatches)

	fmt.Printf("Recall@10: %.4f\n", recall10)
	fmt.Printf("Average server time per batch: %.6f seconds\n", avgBatchTime)

	if err := batchSaveResults(*dataset, *n0, *branch, *beamWidth, *cached, *batchSize, recall10, avgBatchTime); err != nil {
		log.Fatalf("Error saving results: %v", err)
	}
}
