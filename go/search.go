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
	"time"

	"pianopir"
	"sprint/go/utils"
)

type SearchResult struct {
	N0        int     `json:"n0"`
	Branch    int     `json:"branch"`
	BeamWidth int     `json:"beamw"`
	Cached    int     `json:"cached"`
	Recall10  float64 `json:"recall@10"`
	PIRTime   float64 `json:"time"`
}

func paddedEntryByteCount(dim int) uint64 {
	base := uint64(dim) * 4
	return (base + 31) / 32 * 32
}

func levelBatchSize(levelSize int, branch int, beamWidth int) uint64 {
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

	bs := uint64(want)
	if bs < 2 {
		bs = 2
	}
	if bs%2 != 0 {
		bs++
	}

	maxSafe := uint64(levelSize) * 2
	if bs > maxSafe {
		bs = maxSafe
		if bs < 2 {
			bs = 2
		}
		if bs%2 != 0 {
			bs--
			if bs < 2 {
				bs = 2
			}
		}
	}

	return bs
}

func packVectorToWords(vec []float32, entrySizeWords uint64) []uint64 {
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

func dotPacked(words []uint64, xq []float32) float32 {
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

func buildLevelBatchPIRs(xb [][]float32, indices [][]uint32, branch int, beamWidth int, cached int) ([]*pianopir.SimpleBatchPianoPIR, uint64, uint64) {
	if len(xb) == 0 {
		log.Fatalf("empty base vectors")
	}
	dim := len(xb[0])
	entryByteCount := paddedEntryByteCount(dim)
	entrySizeWords := entryByteCount / 8

	levelPIRs := make([]*pianopir.SimpleBatchPianoPIR, len(indices))
	for lvl := 0; lvl < len(indices); lvl++ {
		if lvl < cached {
			levelPIRs[lvl] = nil
			continue
		}
		lvlCount := len(indices[lvl])
		if lvlCount == 0 {
			levelPIRs[lvl] = nil
			continue
		}
		rawDB := make([]uint64, uint64(lvlCount)*entrySizeWords)
		for pos := 0; pos < lvlCount; pos++ {
			globalIdx := indices[lvl][pos]
			vec := xb[globalIdx]
			packed := packVectorToWords(vec, entrySizeWords)
			copy(rawDB[uint64(pos)*entrySizeWords:uint64(pos+1)*entrySizeWords], packed)
		}

		batchSize := levelBatchSize(lvlCount, branch, beamWidth)
		pir := pianopir.NewSimpleBatchPianoPIR(uint64(lvlCount), entryByteCount, batchSize, rawDB, 8)
		pir.Preprocessing()
		levelPIRs[lvl] = pir
	}

	return levelPIRs, entryByteCount, entrySizeWords
}

func queryBatchAll(pir *pianopir.SimpleBatchPianoPIR, batch []uint64) ([][]uint64, time.Duration, error) {
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

	out := make([][]uint64, len(batch))
	queryTime := time.Duration(0)

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
	for idx := range positionsByIdx {
		partitionIdx := idx / cfg.PartitionSize
		if partitionIdx >= cfg.PartitionNum {
			return nil, 0, fmt.Errorf("computed partition out of range: idx=%d partitionIdx=%d PartitionNum=%d PartitionSize=%d", idx, partitionIdx, cfg.PartitionNum, cfg.PartitionSize)
		}
		pendingByPartition[partitionIdx] = append(pendingByPartition[partitionIdx], idx)
	}

	done := make(map[uint64]bool, len(positionsByIdx))
	pendingCount := len(positionsByIdx)

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
			filler := uint64(0)
			if cfg.DBSize > 0 {
				filler = 0
			} else {
				filler = 0
			}
			for i := 0; i < extra; i++ {
				round = append(round, filler)
			}
		}

		if uint64(len(round)) != cfg.BatchSize {
			return nil, queryTime, fmt.Errorf("internal error: constructed round of size %d want %d", len(round), cfg.BatchSize)
		}

		onlineStart := time.Now()
		responses, err := pir.Query(round)
		queryTime += time.Since(onlineStart)
		if err != nil {
			return nil, queryTime, err
		}
		if len(responses) != len(round) {
			return nil, queryTime, fmt.Errorf("BatchPIR.Query response length mismatch: got %d want %d", len(responses), len(round))
		}

		for i, idx := range round {
			posList, ok := positionsByIdx[idx]
			if !ok {
				continue
			}
			if done[idx] {
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

func searchPerQuery(xq []float32, xb [][]float32, indices [][]uint32, beamWidth int, cached int, k int, levelPIRs []*pianopir.SimpleBatchPianoPIR, entrySizeWords uint64, pirTotalTime *time.Duration) []uint32 {
	topkCand := utils.NewTopK(k)

	var candPos []uint32

	for i, indicesI := range indices {
		var distI []float32
		var idxI []uint32
		var distCand []float32

		if i < cached {
			distI = make([]float32, len(indicesI))
			for j := range indicesI {
				dist := float32(0)
				vec := xb[indicesI[j]]
				for d := 0; d < len(xq); d++ {
					dist += xq[d] * vec[d]
				}
				distI[j] = dist
			}

			idxI = make([]uint32, len(indicesI))
			for j := range indicesI {
				idxI[j] = uint32(j)
			}

			for j := 0; j < len(idxI)-1; j++ {
				for jj := j + 1; jj < len(idxI); jj++ {
					if distI[idxI[jj]] > distI[idxI[j]] {
						idxI[j], idxI[jj] = idxI[jj], idxI[j]
					}
				}
			}
			distCand = make([]float32, len(idxI))
			for j := range idxI {
				distCand[j] = distI[idxI[j]]
			}
		} else {
			if len(candPos) == 0 {
				log.Fatalf("non-cached level %d but candPos is empty", i)
			}
			if i == 0 {
				log.Fatalf("non-cached level cannot be 0")
			}
			branchFactor := len(indicesI) / len(indices[i-1])
			if branchFactor <= 0 {
				log.Fatalf("invalid branchFactor at level %d", i)
			}

			idxI = make([]uint32, 0, len(candPos)*branchFactor)
			for _, p := range candPos {
				start := int(p) * branchFactor
				for off := 0; off < branchFactor; off++ {
					idxI = append(idxI, uint32(start+off))
				}
			}

			pir := (*pianopir.SimpleBatchPianoPIR)(nil)
			if i < len(levelPIRs) {
				pir = levelPIRs[i]
			}
			if pir == nil {
				log.Fatalf("missing PIR instance for level %d (cached=%d)", i, cached)
			}

			distI = make([]float32, len(idxI))
			batch := make([]uint64, len(idxI))
			for j := range idxI {
				batch[j] = uint64(idxI[j])
			}

			responses, onlineTime, err := queryBatchAll(pir, batch)
			if err != nil {
				log.Fatalf("BatchPIR.Query(level=%d, n=%d) failed: %v", i, len(batch), err)
			}
			if len(responses) != len(batch) {
				log.Fatalf("BatchPIR.Query(level=%d) response length mismatch: got %d want %d", i, len(responses), len(batch))
			}
			for j := range responses {
				resp := responses[j]
				if uint64(len(resp)) < entrySizeWords {
					log.Fatalf("BatchPIR response too small: got %d words want %d", len(resp), entrySizeWords)
				}
				distI[j] = dotPacked(resp[:entrySizeWords], xq)
			}
			*pirTotalTime += onlineTime
			for j := 0; j < len(idxI)-1; j++ {
				for jj := j + 1; jj < len(idxI); jj++ {
					if distI[jj] > distI[j] {
						distI[j], distI[jj] = distI[jj], distI[j]
						idxI[j], idxI[jj] = idxI[jj], idxI[j]
					}
				}
			}
			distCand = distI
		}

		if beamWidth > 0 && len(idxI) > beamWidth {
			candPos = idxI[:beamWidth]
		} else {
			candPos = idxI
		}

		for j := uint32(0); j < uint32(len(idxI)); j++ {
			globalIdx := indicesI[idxI[j]]
			dist := distCand[j]
			topkCand.Insert(dist, globalIdx)
		}

	}
	return topkCand.GetTopKIndices()
}

func searchAllQueries(xq [][]float32, xb [][]float32, indices [][]uint32, beamWidth int, cached int, k int, levelPIRs []*pianopir.SimpleBatchPianoPIR, entrySizeWords uint64, allPirTotals *[]time.Duration) [][]uint32 {
	results := make([][]uint32, 0, len(xq))
	for _, xqI := range xq {
		pirTotalTime := time.Duration(0)
		res := searchPerQuery(xqI, xb, indices, beamWidth, cached, k, levelPIRs, entrySizeWords, &pirTotalTime)
		*allPirTotals = append(*allPirTotals, pirTotalTime)
		results = append(results, res)
	}
	return results
}

func getPirMeanTimeSeconds(totals []time.Duration) float64 {
	if len(totals) == 0 {
		return 0
	}

	mean := 0.0
	for _, t := range totals {
		mean += t.Seconds()
	}
	return mean / float64(len(totals))
}

func getProjectRoot() string {
	_, filename, _, _ := runtime.Caller(0)
	return filepath.Dir(filepath.Dir(filename))
}

func saveResults(dataset string, n0 int, branch int, beamWidth int, cached int, recall10 float64, pirMeanTimeSeconds float64) error {
	projectRoot := getProjectRoot()
	resultDir := filepath.Join(projectRoot, "result", dataset)

	if err := os.MkdirAll(resultDir, 0755); err != nil {
		return err
	}

	result := SearchResult{
		N0:        n0,
		Branch:    branch,
		BeamWidth: beamWidth,
		Cached:    cached,
		Recall10:  recall10,
		PIRTime:   pirMeanTimeSeconds,
	}

	jsonBytes, err := json.Marshal(result)
	if err != nil {
		return err
	}

	filePath := filepath.Join(resultDir, "pir.jsonl")
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
	beamWidth := flag.Int("beamw", 1, "Beam width for search (number of PIR batch queries)")
	cached := flag.Int("cached", 1, "Cached index depth")

	flag.Parse()

	xb := utils.LoadBase(*dataset)
	xq := utils.LoadQuery(*dataset)
	gt := utils.LoadGroundtruth(*dataset)

	indices := utils.LoadIndices(*dataset, *n0, *branch)

	levelPIRs, _, entrySizeWords := buildLevelBatchPIRs(xb, indices, *branch, *beamWidth, *cached)

	k := 10
	allPirTotals := make([]time.Duration, 0)
	finalResults := searchAllQueries(xq, xb, indices, *beamWidth, *cached, k, levelPIRs, entrySizeWords, &allPirTotals)

	recall10 := utils.MeanNRecallAtK(finalResults, gt, k, k)
	fmt.Printf("Recall@10: %.4f\n", recall10)

	pirMeanTimeSeconds := getPirMeanTimeSeconds(allPirTotals)
	fmt.Printf("Mean total Server time per query: %.6f seconds\n", pirMeanTimeSeconds)

	if err := saveResults(*dataset, *n0, *branch, *beamWidth, *cached, recall10, pirMeanTimeSeconds); err != nil {
		log.Fatalf("Error saving results: %v", err)
	}
}
