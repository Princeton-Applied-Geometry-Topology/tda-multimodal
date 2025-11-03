# Adversarial Compositional Binding Experiment

## Hypothesis

Vision-Language Models (VLMs) form compositional representations where color and shape are bound together. When we create mismatches between image and text, we can:
1. Measure how the compositional structure degrades
2. Identify which layers are sensitive to binding mismatches
3. Compare whether color vs. shape mismatches disrupt the structure differently

## Experiment Design

### Conditions

For each base image (e.g., showing a **red cube**), we create 4 conditions:

1. **MATCHED** (Control): Image = red cube, Text = "red cube"
   - Expected: Strong compositional structure (high H1 persistence)

2. **COLOR MISMATCH**: Image = red cube, Text = "blue cube" (or any other color)
   - Tests: How sensitive is the model to color binding errors?
   - Expected: Reduced compositional structure, especially if model trusts image over text

3. **SHAPE MISMATCH**: Image = red cube, Text = "red sphere" (or any other shape)
   - Tests: How sensitive is the model to shape binding errors?
   - Expected: Reduced compositional structure

4. **BOTH MISMATCH**: Image = red cube, Text = "blue sphere"
   - Tests: Maximum disruption
   - Expected: Strongest degradation of compositional structure

### Dataset Coverage

- **36 base images**: All combinations of 6 colors × 6 shapes
- **Per base image**:
  - 1 matched sample
  - 5 color mismatches (one for each other color)
  - 5 shape mismatches (one for each other shape)
  - 9 both mismatches (strategic subset: 3 colors × 3 shapes)
- **Total samples**: 36 × (1 + 5 + 5 + 9) = **720 samples**

## Analysis Metrics

### Topological Metrics
- **H1 Persistence**: Measures how strongly the torus/cylinder structure (color × shape product space) persists
- **Number of H1 features**: Count of loops in the representation space
- **H0 Persistence**: Connectedness of the space

### Clustering Metrics (Silhouette Scores)
- **Image Color Silhouette**: How well do representations cluster by the *actual* image color?
- **Text Color Silhouette**: How well do representations cluster by the *text prompt* color?
- **Image Shape Silhouette**: Clustering by actual image shape
- **Text Shape Silhouette**: Clustering by text prompt shape

### Key Comparisons
1. **Persistence Disruption**: `matched_persistence - mismatch_persistence`
   - Shows which layers are most disrupted by mismatches
2. **Image vs. Text Clustering**: If image silhouette > text silhouette, model trusts image more
3. **Color vs. Shape Sensitivity**: Which type of mismatch causes more disruption?

## Expected Findings

1. **Matched condition**: Should show peak H1 persistence at the "binding layer" (similar to original experiment)

2. **Color/Shape mismatch**: Should see reduced H1 persistence, especially in later layers where binding occurs

3. **Both mismatch**: Should show strongest disruption

4. **Layer-wise progression**: Early layers may cluster by image features, later layers should reflect the mismatch resolution

5. **Binding layer disruption**: The layer that normally shows peak compositional structure should show reduced persistence under mismatched conditions

## Files

- `generate_adversarial_metadata.py`: Creates all adversarial image-text pairs
- `extract_adversarial_activations.py`: Extracts activations from the model
- `analyze_adversarial_tda.py`: Performs TDA analysis and compares conditions

## Running the Experiment

```bash
# 1. Generate adversarial metadata
python generate_adversarial_metadata.py

# 2. Extract activations (requires GPU, CUDA module)
module load cudatoolkit/12.8
conda activate interp
python extract_adversarial_activations.py

# 3. Analyze with TDA
python analyze_adversarial_tda.py
```

## Output

Results will be in `tda_adversarial_output/`:
- Separate TDA results for each condition
- Comparison plots showing persistence across conditions
- Layer-by-layer statistics for all metrics

