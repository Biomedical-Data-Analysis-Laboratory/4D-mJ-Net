# CT Perfusion is All We Need: 4D CNN Segmentation of Penumbra and Core in Patients With Suspected Ischemic Stroke
## Release v1.0
It contains the code described in the paper "CT Perfusion is All We Need: 4D CNN Segmentation of Penumbra and Core in Patients With Suspected Ischemic Stroke".

### 1 - Abstract
Stroke is the second leading cause of death worldwide, and around 87 % of strokes are ischemic strokes.  
Accurate and rapid prediction techniques for identifying ischemic regions, including dead tissue (core) and potentially salvageable tissue (penumbra), in patients with acute ischemic stroke (AIS) hold great clinical importance, as these regions are used to diagnose and plan treatment. 
Computed Tomography Perfusion (CTP) is often used as a primary tool for assessing stroke location, severity, and the volume of ischemic regions.
Current automatic segmentation methods for CTP typically utilize pre-processed 3D parametric maps, traditionally used for clinical interpretation by radiologists. An alternative approach is to use the raw CTP data slice by slice as 2D+time input, where the spatial information over the volume is overlooked. Additionally, these methods primarily focus on segmenting core regions, yet predicting penumbra regions can be crucial for treatment planning.

This paper investigates different methods to utilize the entire raw 4D CTP as input to fully exploit the spatio-temporal information, leading us to propose a 4D convolution layer in a 4D CNN network.
Our comprehensive experiments on a local dataset of 152 patients divided into three groups show that our proposed models generate more precise results than other methods explored.
Adopting the proposed _4D mJ-Net_, a Dice Coefficient of 0.53 and 0.23 is achieved for segmenting penumbra and core areas, respectively.
Using the entire 4D CTP data for AIS segmentation offers improved precision and potentially better treatment planning in patients suspected of this condition.

### 2 - Link to paper

TBA

### 3 - Dependecies:
```
pip install -r requirements.txt
```

### 4 - How to cite our work
The code is released free of charge as open-source software under the GPL-3.0 License. Please cite our paper when you have used it in your study.
```
TBA
```

### Got Questions?
Email the author at luca.tomasetti@uis.no

