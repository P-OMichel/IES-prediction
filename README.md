# Predicting Isoelectric Suppression 90 Seconds in Advance

This repository contains code and workflows for predicting **isoelectric suppression (IES)** up to **90 seconds in advance** using **EEG feature engineering** focused on **delta** and **alpha** wave dynamics combined with **gradient boosted trees**.

## Overview

- **Target:** Isoelectric suppression (IES) onset/occurrence with a **90-second prediction horizon**.
- **Signals & Features:** Engineered features derived from **delta** and **alpha** EEG frequency bands (such as band power, spectral ratios, temporal trends, and rolling statistics).
- **Modeling:** Gradient boosted decision tree models (e.g., XGBoost, LightGBM, or CatBoost).
