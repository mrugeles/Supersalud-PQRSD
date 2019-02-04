# Predicting patients with life at risk in the Colombian Healthcare System.

This project is a second version of my project [Building a classification model to detect patients with life at risks from the ”Right of request” dataset in the Colombian Health Care system](https://github.com/mrugeles/capstone-project/blob/master/ProjectReport.pdf). In the first version the model had a score of 80%.

For this version CRISP-DM process was applied allowing the model to get a better score of 85%.

A general explanation for this project can be found at medum.com: [Predicting patients with life at risk in the Colombian Healthcare System.](https://medium.com/@mrugeles/predicting-patients-with-life-at-risk-in-the-colombian-healthcare-system-4a260a0ccbd2).

For this project is required to download the dataset from www.datos.gov.co and installing python libraries category_encoders, seaborn and mlxtend

# Datasets

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

# Python libraries
Python libraries category_encoders, seaborn and mlxtend are required.
```sh
$ pip install category_encoders
$ pip install seaborn
$ pip install mlxtend
$ pip install shap
```
# Notebooks
Jupyter notebooks are numbered and should be run in that order.
