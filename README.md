# ToxNet

ToxNet is a data science group from DIRECT program at the University of Washington, which also collaborates with PNNL. The research is to predict the toxicity of various chemicals based on their structures. We utilize the published dataset from U.S Department of Health and Human Services' National Toxicology Program (https://ntp.niehs.nih.gov/pubhealth/evalatm/test-method-evaluations/acute-systemic-tox/models/index.html) and convert those tabular data into image and text file applying Open-Source Cheminformatics Software [RDKit] (http://www.rdkit.org). Furthermore, we fit these different format of data into three different neural networks: RNN (text), CNN (image) and MLP (tabular). Our primary goal is to analyze whether these models provide similar prediction and if there is a model that performs consistently better than others.

---

### Description of Models

#### MLP

#### RNN

#### CNN

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
* pip install numpy
* pip install pandas
* pip install seaborn
* pip install -U scikit-learn
* pip install tensorflow-gpu
    * _follow instructions in tensorflow website to install CUDA and cuDNN (https://www.tensorflow.org/install/install_sources)_
* pip install keras
* pip install rdkit

---

### Acknowledgments

....
