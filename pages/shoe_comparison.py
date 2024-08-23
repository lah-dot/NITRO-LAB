import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def read_files(file_path):
    data = pd.read_excel(file_path, header=2)
    return data

def get_z_scores(data):
    variables = [
        'Weight (g)', 'Rear_Max Deformation (mm)', 'Rear_Stiffness1 (N/mm)', 
        'Rear_Stiffness2 (N/mm)', 'Rear_Energy In (J)', 'Rear_Energy Returned (J)', 
        'Rear_Energy Returned %', 'Rear_Max Deformation %', 'Stiffness (N.m/deg)', 
        'gague forefoot', 'gague midfoot', 'gague rearfoot', 'Stability features'
    ]
    
    z_scores = pd.DataFrame(columns=variables)
    
    for var in variables:
        mean = np.mean(data[var])
        std_dev = np.std(data[var])
        z_scores[var] = (data[var] - mean) / std_dev
    
    z_scores.insert(0, 'Shoe Name', data['Shoe Name'])
    z_scores['Weight (g)'] *= -1  # Inverse weight for "lightness"

    # Cushioning calculation
    cushion_vars = ['Rear_Max Deformation (mm)', 'Rear_Energy In (J)', 'Rear_Stiffness2 (N/mm)']
    cushion_weights = [0.25, 0.5, 0.25]
    z_scores['Cushioning'] = z_scores[cushion_vars].dot(cushion_weights)

    # Stability calculation
    stability_vars = ['gague midfoot', 'gague forefoot', 'gague rearfoot', 
                      'Rear_Max Deformation %', 'Stability features']
    stability_weights = [0.35, 0.05, 0.05, 0.15, 0.6]
    z_scores['Stability'] = z_scores[stability_vars].dot(stability_weights)
    
    return z_scores

def compare_shoes(z_scores, shoe_names, variables):
    values = z_scores[z_scores['Shoe Name'].isin(shoe_names)][variables]
    
    # Calculate pairwise Euclidean distances between shoes
    distance_matrix = pdist(values, metric='euclidean')
    
    # Convert the distance matrix to a square form
    distance_matrix_square = squareform(distance_matrix)
    
    # Create a DataFrame for easier interpretation
    distance_df = pd.DataFrame(distance_matrix_square, index=shoe_names, columns=shoe_names)
    np.fill_diagonal(distance_df.values, 0)
    
    # Find the pair of shoes with the smallest distance
    min_distance = distance_df[distance_df > 0.001].min().min()
    most_similar_pair = distance_df[distance_df == min_distance].stack().index.tolist()
    
    return most_similar_pair, distance_df

def app():
    st.title("Shoe Comparison Tool")

    file_path = 'C:/Users/laura.healey/OneDrive - PUMA/Shared Inno Research/1. Running/2024June_SiloTestStrategy/Silo_testing.xlsx'
    data = read_files(file_path)
    z_scores = get_z_scores(data)

    variables = ['Cushioning', 'Weight (g)', 'Rear_Energy Returned %', 'Stiffness (N.m/deg)', 'Stability']
        
    # Create checkboxes for each shoe
    selected_shoes = []
    for shoe_name in z_scores['Shoe Name']:
        if st.checkbox(shoe_name, key=shoe_name):
            selected_shoes.append(shoe_name)

    if st.button("Compare Shoes"):
        if len(selected_shoes) < 2:
            st.error("Please select at least two shoes to compare.")
        else:
            most_similar_pair, distance_df = compare_shoes(z_scores, selected_shoes, variables)
                
            st.write(f"Most similar pair of shoes: {most_similar_pair} with a distance of {distance_df[distance_df > 0.001].min().min():.2f}")

            # Plotting the pairwise distance matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(distance_df, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
            plt.title("Pairwise Distance Matrix (Euclidean)")
            st.pyplot(plt)

if __name__ == "__main__":
    app()