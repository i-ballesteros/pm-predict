# Forecasting Particulate Matter Count in London

Every spring it's the same story. Days get longer and temperatures rise here in the UK, so I go on longer outdoor runs. However, with spring comes higher pollen and pollution levels, leading to the dreaded [runner’s nose](https://www.emjreviews.com/respiratory/article/exercise-and-rhinitis-in-athletes/) for days after a run. Thankfully this doesn’t happen every day—only when particulate matter spikes. If only I could predict PM2.5 concentrations to plan my run days...

This project demonstrates an end-to-end machine learning pipeline for forecasting hourly PM2.5 concentrations in London.  It leverages:

* [OpenAQ](https://openaq.org/) for historical PM2.5 sensor data
* [Open-Meteo](https://open-meteo.com/) for weather forecasts (temperature, wind, humidity, etc.)

By combining past pollutant readings with weather features, the goal is to predict days when outdoor air quality is poor and plan my runs accordingly.

> **Work in progress:** This repo includes data ingestion, feature engineering, model training, validation, and deployment on Google Cloud Vertex AI.

---

## Project Structure

```
project_root/
│
├── src/                          # Source code
│   ├── data_ingestion.py         # Load & merge OpenAQ + weather data
│   ├── feature_engineering.py    # Time-based & weather feature transforms
│   ├── train.py                  # Train/validate model with proper split
│
├── Dockerfile                    # Dockerfile used to build model image from vertex-ai/training/sklearn-cpu.1-6:latest 
```