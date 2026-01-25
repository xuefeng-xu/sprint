package utils

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strconv"

	"github.com/kshard/fvecs"
)

func getDatasetPath(dataset string) string {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		panic("runtime.Caller failed")
	}

	sourceDir := filepath.Dir(filename)
	projectRoot := filepath.Join(sourceDir, "..", "..")
	return filepath.Join(projectRoot, "dataset", dataset)
}

func LoadFvecs(dataset string, ftype string) [][]float32 {
	datasetDir := getDatasetPath(dataset)
	filename := dataset + "_" + ftype + ".fvecs"
	filePath := filepath.Join(datasetDir, filename)

	r, err := os.Open(filePath)
	if err != nil {
		panic(err)
	}
	defer r.Close()

	var data [][]float32

	d := fvecs.NewDecoder[float32](r)

	for {
		v, err := d.Read()
		if err != nil {
			break
		}
		data = append(data, v)
	}

	return data
}

func LoadIvecs(dataset string, ftype string) [][]uint32 {
	datasetDir := getDatasetPath(dataset)
	filename := dataset + "_" + ftype + ".ivecs"
	filePath := filepath.Join(datasetDir, filename)

	r, err := os.Open(filePath)
	if err != nil {
		panic(err)
	}
	defer r.Close()

	var data [][]uint32

	d := fvecs.NewDecoder[uint32](r)

	for {
		v, err := d.Read()
		if err != nil {
			break
		}
		data = append(data, v)
	}

	return data
}

func LoadBase(dataset string) [][]float32 {
	return LoadFvecs(dataset, "base") // fvecs
}

func LoadQuery(dataset string) [][]float32 {
	return LoadFvecs(dataset, "query") // fvecs
}

func LoadGroundtruth(dataset string) [][]uint32 {
	// k=100 ground truth
	return LoadIvecs(dataset, "gt_ip_k_100") // ivecs
}

func LoadIndices(dataset string, n0 int, branch int) [][]uint32 {
	datasetDir := getDatasetPath(dataset)
	indiceDir := filepath.Join(datasetDir, "indices")

	filename := "n0_" + strconv.Itoa(n0) + "_branch_" + strconv.Itoa(branch) + ".json"
	filePath := filepath.Join(indiceDir, filename)
	data, err := os.ReadFile(filePath)
	if err != nil {
		panic(err)
	}

	var indices [][]uint32
	err = json.Unmarshal(data, &indices)
	if err != nil {
		panic(err)
	}

	return indices
}
