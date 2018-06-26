# ToxNet

ToxNet is a data science group from DIRECT program at the University of Washington, which also collaborates with PNNL. The research is to predict the toxicity of various chemicals based on their structures. We utilize the published dataset from U.S Department of Health and Human Services' National Toxicology Program (https://ntp.niehs.nih.gov/pubhealth/evalatm/test-method-evaluations/acute-systemic-tox/models/index.html) and convert those tabular data into image and text file applying Open-Source Cheminformatics Software [RDKit] (http://www.rdkit.org). Furthermore, we fit these different format of data into three different neural networks: RNN (text), CNN (image) and MLP (tabular). Our primary goal is to analyze whether these models provide similar prediction and if there is a model that performs consistently better than others.

---

### Description of Models

#### MLP

Generally, multilayer perceptron (MLP) performs well to train the tabular form of data. We adopted this strategy for our project. In this network, we tuned the hyperparameters including dropout rate, number of layers (num_layer), activation function (relu_type), number of nodes per layer (layer_units), regulization type (reg_type) and regulization value (reg_value).

<p align="center">
  <b>Table 1: Hyperparameter tunning result for MLP</b><br>
</p>

|               | dropout rate | num_layer | relu_type | layer_units | reg_type | reg_value |
| ------------- | ------------ | --------- | ----------| ----------- | -------- | --------- |
|    nontoxic   | 0.5 | 2 | prelu | 256,32 | l2 | 2.5 |
|    verytoxic  |
|    GHS        |
|    EPA        |
|    LD50       |

#### RNN

We use recurrent neural network (RNN) to train the text infomation in our project. In this case, we tuned the hyperparameters including embedding dimension (em_dim), activation function (relu_type), convolutional units (conv_units), number of layers (num_layer), number of nodes per layer (layer_units), regulization type (reg_type) and regulization value (reg_value).

<p align="center">
  <b>Table 2: Hyperparameter tunning result for RNN</b><br>
</p>

|               |    em_dim    |     relu_type   | conv_units | reg_type | reg_value | num_layer | layer_units |
| ------------- | ------------ | ----------------| ---------- | -------- | --------- | --------- | ----------- |
|    nontoxic   | 7 | prelu | 64 | L2 | 4.5 | 2 | 64 |
|    GHS        | 4 | prelu | 128 | L2 | 5 | 2 | 64 |
|    verytoxic  | 2 | prelu | 64 | L2 | 5 | 2 | 64 |
|    EPA        | 3 | leakyrelu | 64 | L2 | 3.5 | 2 | 64 |
|    LD50       | 4 | prelu | 64 | L2 | 5 | 2 | 32|

#### CNN

Convolutional neural network (CNNN) is designed for image training. In this situation, we did not finish all the parameter tuning, instead, we use a empirical setting for the network (with 64 filters, 3 blocks and no dropout rate). The prediction result for CNN is not as good as MLP and RNN.

---

### Data Visualization

1. Accuracy Comparison

![](https://github.com/Luochenghuang/ToxNet/blob/master/Poster/result_comparison.png)

<p align="center">
  <b>Fig 1. Accuracy Evaluation for All 5 Tasks</b><br>
</p>

NOTE: The left hand figure uses AUC to represent accuracy for all classification tasks and right-hand side RMSE is utilized for regression evaluation. In these figures, the MLP always performs better than other two networks. However, it may be misleading since we did not take all parameters into consideration.


2. Result for Indivdual Task


![](https://github.com/Luochenghuang/ToxNet/blob/master/Poster/confusion_matrix.png) | ![](https://github.com/Luochenghuang/ToxNet/blob/master/Poster/log50_comparison.png)
:----------------------------------------:|:------------------------------------------:
Fig 2. Confusion Matrix for Classification Tasks | Fig 3. Regression Plot for Regression Task



---

### Files Setup
* The functions are in chem_scripts folder.
* The modules folder provides all the executable files. `MLP_Prototype.py` includes code for training, prediction and recording results. The rest of `.py` files are for data cleaning and data extraction.
* The data directory records all the useful data.

---

### Dependencies

* numpy
* pandas
* seaborn
* sklearn
* tensorflow (GPU backend)
    * _You need to install CUDA (>= 7.0, version 9 recommended) and cuDNN (>=v3, version 7 recommended)_
* keras
* RDKit

---

### Installation
* $ pip install numpy==1.14.4
* $ pip install pandas==0.23.1
* $ pip install seaborn==0.8.1
* $ pip install -U scikit-learn==0.19.1
* $ pip install tensorflow-gpu==1.8.0
    * _follow instructions in tensorflow website to install CUDA and cuDNN (https://www.tensorflow.org/install/install_sources)_
* $ pip install Keras==2.2.0
* $ conda create -c rdkit -n my-rdkit-env rdkit (version 2017.09.1)

---

### Acknowledgments

Special thanks to Garrett Goh’s careful and patient guidance on our project and amazing support form  Prof. Jim Pfaendtner and Prof.Dave Beck.
