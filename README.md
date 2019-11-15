# Predicting patients with life at risk in the Colombian Healthcare System.

This project is a second version of my project [Building a classification model to detect patients with life at risks from the ”Right of request” dataset in the Colombian Health Care system](https://github.com/mrugeles/capstone-project/blob/master/ProjectReport.pdf). In the first version the model had a score of 80%.

For this version CRISP-DM process was applied allowing the model to get a better score of 85%.

## Project’s goals
This project aims to answer the following questions:

* **How much is used the internet for raising PQRDs?** As more people has access to internet, it makes sense to assume that more PQRDs will be raised using some internet service like e-mail or web forms.
* **Which are the most common causes for raising a PQRD?** Serious illness such as as cancer or similar would be expected to appear in the analysis.
* **Can we make some predictions with this data?** We’ll explore features of interest related with responding PQRDs and see if any useful predictions can be done with the dataset.

A general explanation for this project can be found at medum.com: [Predicting patients with life at risk in the Colombian Healthcare System.](https://medium.com/@mrugeles/predicting-patients-with-life-at-risk-in-the-colombian-healthcare-system-4a260a0ccbd2).

## Requirements
For this project is required to download the dataset from www.datos.gov.co and installing python libraries category_encoders, seaborn and mlxtend

## Datasets

The downloaded datasets should be placed in subfolder "datasets/". It's posible to downloads the datasets directly from:
  - https://www.datos.gov.co/api/views/36n3-fsjh/rows.csv?accessType=DOWNLOAD
  - https://www.datos.gov.co/api/views/b3xk-8uh2/rows.csv?accessType=DOWNLOAD
  - https://www.datos.gov.co/api/views/gg2r-kx6x/rows.csv?accessType=DOWNLOAD

If the direct links doesn't work, download datasets from these urls:
  - https://www.datos.gov.co/Salud-y-Protecci-n-Social/Base-De-Datos-PQRD-2015/36n3-fsjh
  - https://www.datos.gov.co/Salud-y-Protecci-n-Social/Base-De-Datos-PQRD-2016/b3xk-8uh2
  - https://www.datos.gov.co/es/Salud-y-Protecci-n-Social/Base-De-Datos-PQRD-2017/gg2r-kx6x

In every link, click the button "Exportar" and then click the button "CSV" to download the datasets in csv format and place them in the subfolder "datasets".

[![N|Solid](https://raw.githubusercontent.com/mrugeles/mrugeles.github.io/master/images/downloaddataset.png)](https://raw.githubusercontent.com/mrugeles/mrugeles.github.io/master/images/downloaddataset.png)

### CIE 10 dataset

https://datos.narino.gov.co/?q=dataset/cat%C3%A1logos-de-sector-salud/resource/1b117b20-493a-4965-87c8-b1db15dad898

## Python libraries
Python libraries category_encoders, seaborn and mlxtend are required.
```sh
$ pip install category_encoders
$ pip install seaborn
$ pip install mlxtend
$ pip install shap
```
## Notebooks
Jupyter notebooks are numbered and should be run in that order.
