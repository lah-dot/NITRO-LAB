import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def plot_speed_distance(z_scores):
    # Calculate speed score
    variables = z_scores[['Rear_Energy Returned %', 'Weight (g)', 'Stiffness (N.m/deg)']]
    weights = [0.50, 0.15, 0.35]
    speed = np.dot(variables, weights) / np.sum(weights)
    
    # Calculate distance score
    variables = z_scores[['Cushioning', 'Stability']]
    weights = [0.75, 0.15]
    distance = np.dot(variables, weights) / np.sum(weights)
    
    # Flatten the arrays for plotting
    speed = speed.flatten()
    distance = distance.flatten()
    
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(speed, distance)
    
    # Add labels for each point
    for i, shoe_name in enumerate(z_scores['Shoe Name']):
        plt.text(speed[i], distance[i], shoe_name, fontsize=9, ha='right')
    
    # Set labels and title
    plt.xlabel('Speed Score')
    plt.ylabel('Distance Score')
    plt.title('Speed vs. Distance Scatter Plot')
    
    # Show plot
    plt.grid(True)
    st.pyplot(plt)

def app():
    st.title("Shoe Speed and Distance Comparison")

    file_path = 'C:/Users/laura.healey/OneDrive - PUMA/Shared Inno Research/1. Running/2024June_SiloTestStrategy/Silo_testing.xlsx'
    data = read_files(file_path)
    z_scores = get_z_scores(data)

    st.write("Displaying Speed vs. Distance scatter plot...")
    plot_speed_distance(z_scores)

# Run the Streamlit app
if __name__ == "__main__":
    app()