# Requirements
- tensorflow 1.15.0
- numpy 1.16.2

# Inferring ultrasonic resonance frequencies

A deep neural network (DNN) structure is proposed for inferring the resonance frequencies of piezoelectric rectangular parallelepipeds. The performance of the proposed DNN structure was demonstrated by computing the resonance frequencies of PZT-8 and LiNbO3 samples.

### Datasets

PZT-8 and LiNbO3 samples with diverse parameters, including elastic stiffness constants, piezoelectric stress constants, dielectric constants, density, and dimensions, were randomly generated. 200 resonance frequencies of the piezoelectric samples were calculated using the Rayleigh-Ritz method.

### Training & Testing

```bash
python rus_fit_inv.py
```

# Data regression for chemical data

### Datasets
[Chemical data](https://yarpiz.com/263/ypml113-gmdh)

### Training & Testing

```bash
python dnn_regression.py
```

### Results

 Method          | GMDH             | Conventional DNN | Proposed DNN
---------------- | ---------------- | -----------------|------------------
 MSE             | 5.80             | 4.82             | 4.23   
 RMSE            | 2.41             | 2.19             | 2.06

To calculate the MSE/RMSE score of the Group Method of Data Handling (GMDH) on the chemical data, download the [matalb code](https://yarpiz.com/263/ypml113-gmdh), and run this MATLAB script
