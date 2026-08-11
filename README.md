# 🧭 Predictive Tourism – Demand & Resource Optimization

Predictive Tourism is an interactive tourism analytics and machine-learning application developed using Python and Streamlit. The application analyzes tourism demand, provides demand forecasting, compares multiple machine-learning algorithms, supports resource planning, and presents sustainability and community-related insights through an interactive dashboard.

## 📌 Project Overview

The project is designed to help analyze tourism demand and support better planning of tourism resources such as rooms, guides, and vehicles.

The application uses tourism-related data including destination, date, weather conditions, marketing spend, social sentiment, price index, bookings, lead time, and available resources.

It provides an interactive dashboard where users can filter tourism data by destination, date range, and booking channel and explore different analytical visualizations.

## 🎯 Objectives

- Analyze tourism booking patterns and demand.
- Forecast future tourism demand using machine learning.
- Compare different machine-learning algorithms.
- Identify factors affecting tourism demand.
- Analyze resource requirements for tourism destinations.
- Provide sustainability-related tourism insights.
- Present tourism data through interactive visualizations.
- Allow users to upload CSV datasets for analysis.

## 🚀 Key Features

### 1. Overview Dashboard

The application provides an interactive dashboard containing multiple tourism visualizations, including:

- Daily bookings trend
- Rolling 30-day average bookings
- Monthly bookings
- Bookings distribution by weekday
- Destination-wise booking share
- Bookings by channel
- Lead-time distribution
- Price vs. bookings
- Social sentiment vs. bookings
- Temperature vs. bookings
- Feature correlation
- Average party size by destination
- Destination map
- Rain impact on bookings
- Room utilization by destination

The dashboard also displays important KPIs such as total bookings, average price index, average social sentiment, and percentage of rainy days.

## 🤖 2. AI/ML Demand Forecasting

The project uses a **Random Forest Regressor** to forecast future tourism demand.

The forecasting module uses features such as:

- Weekend indicator
- Month
- Day of week
- Temperature
- Rain
- Marketing spend
- Social sentiment
- Price index
- Lead time
- Day of year
- Week of year
- Year
- Previous booking values
- Rolling booking averages

The user can select a destination and choose a forecast horizon between 7 and 90 days.

The model performance is evaluated using:

- Mean Absolute Error (MAE)
- R² Score

The application also provides a comparison between actual and predicted bookings.

## 🧪 3. Machine Learning Model Lab

The application provides a model laboratory to train and compare five machine-learning algorithms:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. K-Nearest Neighbors (KNN)

The models are evaluated using:

- Mean Absolute Error (MAE)
- R² Score

The application identifies the best-performing model based on MAE and provides:

- Model comparison
- MAE comparison chart
- Prediction parity plot
- Residual analysis
- Feature importance for tree-based models

## 🏨 4. Resource Planner

The Resource Planner helps estimate tourism resource requirements based on average demand.

It analyzes the requirement and availability of:

- Hotel rooms
- Tour guides
- Vehicles

The application calculates:

- Average demand
- Rooms required
- Guides required
- Vehicles required
- Room capacity gap
- Guide capacity gap

It also provides a capacity-versus-required visualization.

## 🌱 5. Sustainability Analysis

The Sustainability section provides tourism-related environmental insights.

It estimates:

- Daily emissions
- CO₂ emissions per booking
- Eco score by destination

The application also provides recommendations such as promoting off-peak travel, pooled transportation, and eco-friendly stays to reduce estimated emissions.

## 👥 6. Community Engagement

The Community section demonstrates how local communities can participate in tourism planning.

Users can submit ideas related to:

- Training programs
- Local crafts
- Local tours
- Eco-drives

The application also provides an estimated community revenue-share visualization.

## 📊 Data Processing

The application uses Pandas and NumPy for data processing and transformation.

The project uses tourism data containing fields such as:

- Date
- Destination
- Latitude and longitude
- Weekend indicator
- Month
- Day of week
- Temperature
- Rain
- Marketing spend
- Social sentiment
- Price index
- Bookings
- Average party size
- Booking channel
- Lead time
- Available rooms
- Available guides
- Available vehicles

The application also supports CSV upload through the Streamlit interface. Uploaded datasets are validated and processed before being stored in the SQLite database.

## 🗄️ Database

SQLite is used as the database for storing tourism data.

The project contains:

`tourism.db`

The application automatically checks the database and creates/regenerates the tourism dataset when required.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Web Framework/UI:** Streamlit
- **Database:** SQLite
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Plotly
- **Machine Learning:** Scikit-learn
- **Version Control:** Git and GitHub

## 📁 Project Structure

```text
predictive-tourism/
│
├── app.py
├── database.py
├── style.css
├── tourism.db
├── tourism_10k.csv
├── tourism_rf_strong_10k.csv
└── README.md
