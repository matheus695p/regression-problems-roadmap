# Roadmap for regression problems
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

A roadmap for regressions problems including:
  - feature selection:  xgboost importance and correlation of features
  - preprocesing: conection with S3 (AWS) to get and preprocess data
  - synthetic data generation: Synthetic data generation with VAE's and synthetic minority oversampling techniche (smothe)
  - regressions: linear regressions, decission trees, support vector regressor, XGBoost, NN (with and without PCA or Kernel PCA before the training, customs loss functions and activations), CNN architectures, syntethic 
  - finnetuning: tunner search to find the best architecture on hyper models

### SETUP 
Pasos a seguir para comenzar a etiquetar

```sh
$ git clone https://github.com/matheus695p/regression-problems-roadmap.git
$ cd regression-problems-roadmap
$ echo instalar los requirements
$ pip install -r requirements.txt
```

```sh
│   .gitignore
│   README.md
│
├───fine_tuning
│       Best_Architectures_resultados.txt
│       best_model.png
│
├───images
│    *.png
├───src
│   ├───cnn_architectures
│   │       cnn_architecture.py
│   │
│   ├───feature_selection
│   │       feature_selection.py
│   │
│   ├───fine_tunning_models
│   │       fine_tuning_models.py
│   │       fine_tuning_module.py
│   │
│   ├───nn_architectures
│   │       main_retrain.py
│   │       module_main.py
│   │       training_module.py
│   │
│   ├───preprocessing
│   │       get_snaphots.py
│   │       get_historical_ads.py
│   │       labeller.py
│   │       labeller_module.py
│   │       preprocessing_historical.py
│   │       preprocessing_snapshots.py
│   │
│   └───synthetic_data
│           autoencoder_synthetic_generation.py
│           generate_synthetic_data.py
├───synthetic_data
│       *.png
└───training_results
        Arquitectura 1_ CNN_models.png
        Arquitectura 1_ NN_models.png
        Arquitectura 1_ PCA_NN_models.png
        Arquitectura 2_ CNN_models.png
        Arquitectura 2_ NN_models.png
        Arquitectura 2_ PCA_NN_models.png
        Arquitectura 3_ CNN_models.png
        Arquitectura 3_ NN_models.png
        Arquitectura 3_ PCA_NN_models.png
        Arquitectura 4_ CNN_models.png
        Arquitectura 4_ NN_models.png
        Arquitectura 4_ PCA_NN_models.png
        Arquitectura 5_ CNN_models.png
        Arquitectura 5_ NN_models.png
        Arquitectura 5_ PCA_NN_models.png
        Arquitectura custom loss function_ Final_model.png
        CNN_ CNN_final_model.png
```
