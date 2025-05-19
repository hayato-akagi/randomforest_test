package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
)

type Tree struct {
	ChildrenLeft  []int       `json:"children_left"`
	ChildrenRight []int       `json:"children_right"`
	Feature       []int       `json:"feature"`
	Threshold     []float64   `json:"threshold"`
	Value         [][]float64 `json:"value"`
}

type ScalerParams struct {
	Mean  []float64 `json:"mean"`
	Scale []float64 `json:"scale"`
}

func predictTree(tree Tree, sample []float64) ([]float64, error) {
	node := 0
	for tree.ChildrenLeft[node] != -1 {
		feature := tree.Feature[node]
		if feature >= len(sample) {
			return nil, fmt.Errorf("feature index %d out of range (sample length: %d)", feature, len(sample))
		}
		if sample[feature] <= tree.Threshold[node] {
			node = tree.ChildrenLeft[node]
		} else {
			node = tree.ChildrenRight[node]
		}
	}
	// Return the probability distribution at the leaf node
	// sklearn stores values as [[class0_count, class1_count, ...]]
	// We need to normalize these to get probabilities
	values := tree.Value[node] // Get the values at this node
	total := 0.0
	for _, v := range values {
		total += v
	}
	
	if total == 0 {
		return nil, fmt.Errorf("no samples in leaf node")
	}
	
	probabilities := make([]float64, len(values))
	for i, v := range values {
		probabilities[i] = v / total
	}
	return probabilities, nil
}

// Unfixed process: Majority voting from hard predictions (might differ from Python)
func predictForestUnfixed(forest []Tree, sample []float64) (int, error) {
	// Count votes from each tree
	votes := make(map[int]int)
	
	for i, tree := range forest {
		probabilities, err := predictTree(tree, sample)
		if err != nil {
			return 0, fmt.Errorf("error in tree %d: %v", i, err)
		}
		
		// Find class with highest probability for this tree
		maxProb := 0.0
		maxIdx := 0
		for j, prob := range probabilities {
			if prob > maxProb {
				maxProb = prob
				maxIdx = j
			}
		}
		
		votes[maxIdx]++
	}
	
	// Find class with most votes
	maxVotes := 0
	prediction := 0
	for class, vote := range votes {
		if vote > maxVotes {
			maxVotes = vote
			prediction = class
		}
	}
	
	return prediction, nil
}

// Fixed process: Probability averaging to match Python sklearn behavior
func predictForestFixed(forest []Tree, sample []float64) (int, error) {
	if len(forest) == 0 {
		return 0, fmt.Errorf("empty forest")
	}
	
	// Get first tree probabilities to determine number of classes
	firstProbs, err := predictTree(forest[0], sample)
	if err != nil {
		return 0, fmt.Errorf("error getting first tree probabilities: %v", err)
	}
	
	numClasses := len(firstProbs)
	classProbabilities := make([]float64, numClasses)
	
	// Average probabilities from all trees (sklearn approach)
	for i, tree := range forest {
		treeProbabilities, err := predictTree(tree, sample)
		if err != nil {
			return 0, fmt.Errorf("error in tree %d: %v", i, err)
		}
		
		if len(treeProbabilities) != numClasses {
			return 0, fmt.Errorf("tree %d has %d classes, expected %d", 
				i, len(treeProbabilities), numClasses)
		}
		
		for j, prob := range treeProbabilities {
			classProbabilities[j] += prob
		}
	}
	
	// Average across all trees
	for i := range classProbabilities {
		classProbabilities[i] /= float64(len(forest))
	}
	
	// Return class with highest average probability
	maxIdx := 0
	maxProb := classProbabilities[0]
	for i := 1; i < len(classProbabilities); i++ {
		if classProbabilities[i] > maxProb {
			maxProb = classProbabilities[i]
			maxIdx = i
		}
	}
	
	return maxIdx, nil
}

func applyStandardScaler(sample []float64, params ScalerParams) []float64 {
	scaled := make([]float64, len(sample))
	for i := range sample {
		scaled[i] = (sample[i] - params.Mean[i]) / params.Scale[i]
	}
	return scaled
}

func main() {
	// Load model
	data, err := os.ReadFile("/data/rf_model.json")
	if err != nil {
		log.Fatal(err)
	}
	var forest []Tree
	if err := json.Unmarshal(data, &forest); err != nil {
		log.Fatal(err)
	}

	// Load scaler parameters
	scalerData, err := os.ReadFile("/data/scaler.json")
	if err != nil {
		log.Fatal(err)
	}
	var scaler ScalerParams
	if err := json.Unmarshal(scalerData, &scaler); err != nil {
		log.Fatal(err)
	}

	// Load test.csv
	file, err := os.Open("/data/test.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	if len(records) < 1 {
		log.Fatal("CSV is empty")
	}
	header := records[0]
	rows := records[1:]

	// Identify feature column indices
	var featureIndices []int
	for i, name := range header {
		if name != "label" && name != "python_label" && name != "go_label" && name != "fixed_go_label" {
			featureIndices = append(featureIndices, i)
		}
	}

	// Check if go_label and fixed_go_label columns already exist
	hasGoLabel := false
	hasFixedGoLabel := false
	for _, name := range header {
		if name == "go_label" {
			hasGoLabel = true
		}
		if name == "fixed_go_label" {
			hasFixedGoLabel = true
		}
	}

	// Add new columns only if they don't exist
	if !hasGoLabel {
		header = append(header, "go_label")
	}
	if !hasFixedGoLabel {
		header = append(header, "fixed_go_label")
	}

	for i, row := range rows {
		var sample []float64
		for _, j := range featureIndices {
			v, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				log.Fatalf("Failed to parse float at row %d col %d: %v", i+1, j, err)
			}
			sample = append(sample, v)
		}

		// Original prediction (unscaled) - using unfixed majority voting
		goLabel, err := predictForestUnfixed(forest, sample)
		if err != nil {
			log.Fatalf("Error predicting with unfixed method at row %d: %v", i+1, err)
		}

		// Scaled prediction (compensated) - using fixed probability averaging
		scaledSample := applyStandardScaler(sample, scaler)
		fixedLabel, err := predictForestFixed(forest, scaledSample)
		if err != nil {
			log.Fatalf("Error predicting with fixed method at row %d: %v", i+1, err)
		}

		// Update or append the prediction columns
		if hasGoLabel && hasFixedGoLabel {
			// Both columns exist, update them
			for j, name := range header {
				if name == "go_label" {
					row[j] = strconv.Itoa(goLabel)
				}
				if name == "fixed_go_label" {
					row[j] = strconv.Itoa(fixedLabel)
				}
			}
		} else {
			// Add new columns
			if !hasGoLabel {
				row = append(row, strconv.Itoa(goLabel))
			}
			if !hasFixedGoLabel {
				row = append(row, strconv.Itoa(fixedLabel))
			}
		}
		rows[i] = row
	}

	// Write back to test.csv
	outFile, err := os.Create("/data/test.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer outFile.Close()
	writer := csv.NewWriter(outFile)
	defer writer.Flush()

	writer.Write(header)
	for _, row := range rows {
		writer.Write(row)
	}

	// Summary
	pythonIdx := -1
	goIdx := -1
	fixedIdx := -1

	for i, col := range header {
		if col == "python_label" {
			pythonIdx = i
		}
		if col == "go_label" {
			goIdx = i
		}
		if col == "fixed_go_label" {
			fixedIdx = i
		}
	}

	if pythonIdx == -1 {
		log.Println("⚠ Warning: No python_label column found for comparison.")
		return
	}

	var correctGo, correctFixed int
	total := len(rows)

	for _, row := range rows {
		if goIdx != -1 && row[pythonIdx] == row[goIdx] {
			correctGo++
		}
		if fixedIdx != -1 && row[pythonIdx] == row[fixedIdx] {
			correctFixed++
		}
	}

	log.Println("📊 Summary Report:")
	if goIdx != -1 {
		log.Printf("➡  go_label vs python_label      : %d / %d correct (%.2f%%)", correctGo, total, float64(correctGo)*100/float64(total))
	}
	if fixedIdx != -1 {
		log.Printf("✅ fixed_go_label vs python_label: %d / %d correct (%.2f%%)", correctFixed, total, float64(correctFixed)*100/float64(total))
	}
}