# Requirements
- tensorflow 1.15.0
- numpy 1.16.2

# Predicting ultrasonic resonance frequencies

A deep neural network (DNN) structure is proposed for predicting the resonance frequencies of piezoelectric rectangular parallelepipeds. The performance of the proposed DNN structure was demonstrated by computing the resonance frequencies of PZT-8 and LiNbO3 samples.

### Datasets

PZT-8 and LiNbO3 samples with diverse parameters, including elastic stiffness constants, piezoelectric stress constants, dielectric constants, density, and dimensions, were randomly generated. 200 resonance frequencies of the piezoelectric samples were calculated using the Rayleigh-Ritz method.

### Training & Testing

```bash
python rus_fit_inv.py
```

### Experimental results

The performance of the proposed DNN with D{320, 320, 320, 320, 320, 320} was also evaluated. For the test samples, the relative errors (RE) between the resonance frequencies obtained using the Rayleigh-Ritz method and those obtained using the DNN were calculated. For the LiNbO3 samples, the mean and SD of the relative errors attained by the DNN were 0.0937 and 0.080, respectively. For the PZT8 samples, the mean and SD of the relative errors by the DNN were 0.123 and 0.107, respectively. Moreover, the number of the resonance frequencies in each interval was calculated according to their relative errors by dividing the relative error into six intervals, namely [0, 0.05), [0.05, 0.1), [0.1, 0.2), [0.2, 0.5), [0.5, 1.0), and [1.0, ∞). The proportions of resonance frequencies with relative errors at different intervals are listed in the table below. 

Proportion of resonance frequencies with relative error in different intervals
 RE (%)     | LiNbO3        | PZT-8
----------- | ------------- | --------------
 <0.05      | 35.4          | 27.4             
 0.05–0.1   | 27.5          |	23.8           
 0.1–0.2    |	27.2          |	29.8
 0.2–0.5    |	9.6           |	18.2
 0.5–1.0    |	0.2           |	0.7
 RE>1.0     |	0.0           |	0.0


# Data regression for chemical data

Use the proposed DNN structure to process the [Chemical data](https://yarpiz.com/263/ypml113-gmdh).

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

To calculate the MSE/RMSE score of the Group Method of Data Handling (GMDH) on the chemical data, download the [matalb code](https://yarpiz.com/wp-content/uploads/2015/09/ypml113-gmdh.zip), and run this MATLAB script
