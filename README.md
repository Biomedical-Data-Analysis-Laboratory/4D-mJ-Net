# Exploiting 4D CT Perfusion for segmenting infarcted areas in patients suspected of acute ischemic stroke 
## Release v1.0
It contains the code described in the paper "Exploiting 4D CT Perfusion for segmenting infarcted areas in patients suspected of acute ischemic stroke ".

### 1 - Abstract
Precise and fast prediction methods for ischemic areas (core and penumbra) in acute ischemic stroke (AIS) patients are of extreme clinical interest.
Computed Tomography (CT) scan is one of the primary modalities for early assessment in patients suspected of AIS.
CT Perfusion (CTP) is often used as a primary assessment to determine stroke location, severity, and volume lesions.
Precise segmentation of ischemic areas plays an essential role in improving diagnosis and treatment planning.
Current deep neural network methods mostly use colormaps or the 2D+time CT data, which provide limited information as input, leading to less precise results. 
We propose three approaches that rely on the whole 4D CTP study as input.
Additionally, we introduce a novel 4D Convolution layer to work with the entire 4D CTP scan in input to exploit the spatio-temporal information effectively in the studies.
Our comprehensive experiments on a private dataset comprised of 152 patients divided into three groups show that our proposed models generate more precise results than other methods explored.
A Dice Coefficient of 0.70 and 0.45 is achieved for penumbra and core areas, respectively.

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
Email me at luca.tomasetti@uis.no

