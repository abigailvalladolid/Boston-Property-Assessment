#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS5110; Final Project
Properties Total Value Predictive Model
@author: Catherina Haast and Abigail Valladolid
"""

# Necessary libraries, tables and code found on repository
import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import scipy.spatial.distance as ssd
from password_handler import PasswordHandler
import os



def filter_data(city, bd_rms, fl_bth, hlf_bth, park, db_connection):
    """
    Filter data based on user inputs using an SQL query.
    """
    query = """
    SELECT ps.living_area, r.tt_rooms, v.total_value
    FROM property AS p
    JOIN rooms AS r ON p.p_id = r.p_id
    JOIN amenities AS a ON p.p_id = a.p_id
    JOIN property_sf AS ps ON p.p_id = ps.p_id
    JOIN value AS v ON p.p_id = v.p_id
    WHERE city = %s
    AND bed_rms = %s
    AND full_bth = %s
    AND half_bth = %s
    AND num_parking = %s
    """
    params = (city, bd_rms, fl_bth, hlf_bth, park)

    filtered_df = pd.read_sql(query, con=db_connection, params=params)
    return filtered_df

def accurate(pred, original):
    """
    Calculate the accuracy of the predictions, calculates within a $75,000 price range.
    """
    count = 0
    dif = [abs(price - original[idx]) for idx, price in enumerate(pred)]
    
    within_range = sum(1 for diff in dif if diff <= 75000)
    accuracy = within_range / len(dif)
    return accuracy

def normalize(lst):
    """
    Min-max normalization for a list of numbers.
    """
    mn, mx = min(lst), max(lst)
    return [(num - mn) / (mx - mn) for num in lst]

def main():
    """
    Main function for the predictive model and visualization.
    """
    
    password = os.getenv("MYSQL_PASSWORD")
    
    # If password is None, raise an error
    if password is None:
        raise ValueError("MYSQL_PASSWORD is not set")
    
    # Connect to MySQL database
    db_connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password=password,
        database='property_data'
    )

    try:
        '''
        This is where the user can denote property features
        '''
        
        city = input("What city would you like to look at?").upper()
        bd_rms = int(input("How many bedrooms are you looking for?"))
        fl_bth = int(input("How many full baths would you like?"))
        hlf_bth = int(input("How many half baths are you looking for?"))
        park = int(input("How many parking spaces are you looking for?"))

        # Filter data using SQL query
        filtered_df = filter_data(city, bd_rms, fl_bth, hlf_bth, park, 
                                  db_connection)

        if filtered_df.empty:
            print("No matching properties found.")
            return

        print(filtered_df)

        # Split the DataFrame into features (X) and target variable (y)
        X = filtered_df.drop('total_value', axis=1)
        y = filtered_df['total_value']
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.2, 
                                                            random_state=0)
        
        # Find the best k
        k_values = list(range(4, 11))
        k_accuracies = {}

        for k in k_values:
            # Initialize KNN classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            # Train the model
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)
            y_test_values = y_test.tolist()

            acc = accurate(y_pred, y_test_values)
            k_accuracies[k] = acc
            
        
        best_k = max(k_accuracies, key=k_accuracies.get)
        best_accuracy = k_accuracies[best_k]
        print(f"Best k: {best_k}, Accuracy: {best_accuracy:.2f}")

        # Additional inputs for visualization
        sqft = int(input(
            "What is the approximate square footage you are looking for?"))
        ttrms = int(input(
            "What is the total number of rooms you are looking for?"))

        # Normalize inputs and existing data
        la_lst = filtered_df["living_area"].tolist()
        tt_lst = filtered_df["tt_rooms"].tolist()

        new_sqft = (sqft - min(la_lst)) / (max(la_lst) - min(la_lst))
        new_ttr = (ttrms - min(tt_lst)) / (max(tt_lst) - min(tt_lst))

        norm_la = normalize(la_lst)
        norm_tt = normalize(tt_lst)

        # Scatter plot visualization
        plt.scatter(x=norm_la, y=norm_tt)
        plt.xlabel("Living Area")
        plt.ylabel("Total Rooms")
        plt.plot(new_sqft, new_ttr, color="RED", marker="X")
        plt.show()

        # Calculate distances and predict value using weighted average
        all_distances = [
            ssd.euclidean([norm_la[i], norm_tt[i]], [new_sqft, new_ttr])
            for i in range(len(norm_la))
        ]

        closest_distances = sorted(all_distances)[:best_k]
        total_values = filtered_df["total_value"].tolist()

        weights = [1 - dist for dist in closest_distances]
        predicted_value = sum(
            weight * total_values[all_distances.index(dist)]
            for weight, dist in zip(weights, closest_distances)
        ) / sum(weights)

        print(f"Predicted Property Value: ${round(predicted_value):,}")

    finally:
        # Close connection to DB
        db_connection.close()

if __name__ == "__main__":
    main()
