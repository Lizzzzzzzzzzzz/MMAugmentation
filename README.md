# Mathematical Morphology Augmentation - Supplementary Code

This repository contains the implementation of the mathematical morphology augmentation technique described in the manuscript under review. This code is provided as supplementary material to facilitate reproducibility of the results.

## Method Overview

The proposed method uses morphological operations to create variations of binary images while preserving their structural characteristics.

### Core Operations

- **Erosion**: Expands black regions/shrinks white regions
- **Dilation**: Expands white regions/shrinks black regions
- **Opening**: Removes small white spots
- **Closing**: Fills small black holes

### Augmentation Process

1. **Input**: Converts grayscale images to binary
2. **Target Generation**: Creates variation targets using normal distribution around original ratio
3. **Adaptive Morphology**:
   - Selects operations based on current vs. target ratio
   - Applies operations at random points with varying intensity
4. **Output**: Crops images to remove edge artifacts and saves variations

## Implementation Details

The Jupyter notebook (`morphology_augmentation.ipynb`) contains the complete implementation of the method, including:

- Data preprocessing functions
- Implementation of the core morphological operations
- The adaptive augmentation algorithm
- Visualization of results
- Experimental setup used in the manuscript

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib (for visualization)

## Usage Instructions

### Configuration

```python
# Set your folders
input_folder = 'Your_Input_Folder'
output_folder = 'Your_Output_Folder'
```

The implementation processes images (GI1.PNG-GI4.PNG), generating 500 variations for each, saved as HNP1.PNG-HNP2000.PNG.

### Running the Code

The notebook is designed to be self-contained and can be executed cell by cell to reproduce the results presented in the manuscript. All parameters are set to the values used in the experiments described in the paper.

## Note to Reviewers

This code is provided for review purposes as supplementary material. All experimental results reported in the manuscript can be reproduced using this implementation. If you encounter any issues or have questions regarding the implementation, please contact the editorial office who can relay your questions to the authors while maintaining anonymity.
