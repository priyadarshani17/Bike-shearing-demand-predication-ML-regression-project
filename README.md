# Bike-shearing-demand-predication-ML-regression-project
This project will be performed with the help of Supervised Linear Regression model. We will be performing basic check on the data for any errors followed by EDA and at last we will train our Machine Learning model on the data to predict outputs for new inputs provided.

Problem statement
Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes

Introducation
We have been provided with the bike rental demand data of 2017 December to November 2018 (1 year total), in order to perform supervised liner regression and identify the relationships between different variables. The data includes information on the number of bike rentals, the weather conditions, and the day of the week. The goal is to use this data to predict the demand for bike rentals in the future.

Feature Description
Date - Date on which the bike was rented(year-month-day).

Rented Bike count - Number of bikes rented in that hour.

Hour - Hour of the day.

Temperature -Temperature at that time (Celsius).

Humidity - Percentage of humidity in air (%).

Windspeed - Speed of the wind (m/s).

Visibility - How far is the visibility (*10m).

Dew point temperature - The dew point is the temperature the air needs to be cooled at constant pressure to in order to achieve a relative humidityof 100% (°C).

Solar radiation - Solar radiation is the energy recieved on an area on earth from the Sun (MJ/m²).

Rainfall - Measure of Rainfall (mm).

Snowfall - Measure of Snowfall (cm).

Seasons - which season is bike rented (Winter, Spring, Summer, Autumn).

Holiday - Was it a holiday or not (Holiday/No holiday).

Functional Day - Was it a Functional day or not.(Yes/No).
conclusion:
EDA insights:
1. Rented Bike count contains outliers. Wind Speed, Solar Radiation, Rainfall, and Snowfall contains outliers which wont be a problem.

2. We have more bookings on weekdays combined as compsred to weekend.

3. On Weekdays on average 901325 bikes are rented. On Weekends on average 832843 bikes are rented.

4. Top 3 months where most bikes were rented are June, July and May. Peak periods when maximum bikes were rented is from May to October. The month in which least bikes were rented is January(150006) followed by February(151833) and December(185330).

5. Maximum number of bikes rented on average is in the 18th hour followed by 19th hour and 17th hour. Peak period for bike rented count is from15th hour to 22nd hour, there is a slight increment in the 8th hour also. 
Minimum bikes were rented in 4th and 5th hour. 

6. We can see that no bikes were rented on Non-functioning day. 

7. We can see that there are very high demand for bike on rent in summer season , followed by autumn. 
Least numbers of bike were rented in winter season, that maybe because of cold and snow. 

8. In Summer(green), Peak is at 18th hour and least is at 4th and 5th hour, and it has highest number of rented bike. 
In Autumn(blue), Peak is at 18th hour and least is at 4th and 5th hour. 
In Spring(yellow), Peak is at 18th hour and least is at 4th and 5th hour. 
In winter(red), Peak is at 18th hour and least is at 4th and 5th hour, it has least number of bikes rented. 
We see almost similar trent in Autumn, Summer and Spring. There values are close to each other. 

9. As the Snowfall increases, the bike rent count decreases. 

10. As the Rainfall(mm) increases, the bike rent count decreases. 

11. As the Solar Radiation(MJ/m²) increases, the bike rent count decreases rapidly(log scale). 

12. We can see only at an optimal temperature maximum number of bikes were rented. 
The peak period is between temperature 15°C to 28°C. 
Temperature at which maximum bikes were rented is 23.4°C.  
Temperature at which minimum bikes were rented is -16.9°C. 

13. We can see only at an optimal Humidity percentage, maximum number of bikes were rented. 
Most appropriate humidity percentage is between 30% to 80%. 
Humidity at which maximum bikes were rented is 43%. 
Humidity at which minimum bikes were rented is 10%.   

14. We can see as the visibility increases so the number of bike which are rented. 
Visibility at which maximum bikes were rented is 20000 Meters. 

15. We can see as the Wind Speed(m/s) increases the number of bike which are rented decreases. 
Wind Speed(m/s) at which maximum bikes were rented is 1.4 m/s. 
Wind Speed(m/s) at which minimum bikes were rented is 6.9 m/s. 

16. We have only 3.5% of total rented bikes count on Holiday, majority of bikes were rented on working days(not-holiday). 

17. Maximum bikes were rented on Friday(15.4%), On second place we have Wednesday(15.0%) and than Monday(14.8%). 

ML conclusion:**We performend model training on diffrent models such as :-**

1)Linear Regression

2)Lasso Regression

3)Ridge Regression

4)Elasticnet Regression

5)Polynomial Regression

6)Decision Tree Regression

7)Random Forest Regression

8)Gradient Boosting Regression

9)Xtreme GB Regression

1. Most important features for decision tree regressor are temperature, functioning-day-No and humidity.

2. Most important features for Random forest are temperature, humidity, functioning-day-Yes and functioning-day-No.

3. Most important features for Gradient Boosting are temperature, humidity, functioning-day-Yes and functioning-day-No.

4. Most important features for eXtreme Gradient boosting are functioning-day-No, season_winter and hour_18.

5. We used GridSearchcv on many algorithms for best parameter values and results.

6. We encoded many categorical columns for better model learning. the method used was One-hot encoding.

7. We used squareroot transformation to deal with skewness and also outliers in rented_bike_count, wind_speed, solar_radiation, rainfall and snowfall. as log transformation was increasing the skewness drastically.

8. VIF was high for temperature and dew_point_temperature. so we deleted dew_point_tempoerature feature to avoid multicolleniarity.

9. All numeric Columns are correlated to Rented bike count columns and most of them are positively and few are negatively.
Only snowfall, humidity and rainfall are negatively correlated.
Temperature and Dewpoint temperature are more correlated to rented_bike_count column.

10. All models performed slightly better on Train data as compared to test data. Best Adjusted R2 score we got for test data was 0.928718(by Xtreme_GB) while for train data it was 1(by Dicision_tree) and 0.999828(by Gradient_Boosting)

11. The best accuracy algorithm for our test dataset is Xtreme Gradient boosting algorithm with Adj_R2 score of 0.928718.
We have other algorithm such as polynomial, Gradient boosting, Random forest who also have optimal Adj_R2 score.

12. The best accuracy algorithm for our train dataset is Decision tree algorithm with Adj_R2 score of 1.
We have other algorithm such as Gradient boosting, Xtreme_GB and polynomial who also have optimal Adj_R2 score.

13. The model with least Adjusted R2 score was Ridge regressor for both train and test data.

14. We also used crossvalidation for better model performance.
