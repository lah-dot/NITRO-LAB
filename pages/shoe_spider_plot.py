#page 1
# pages/page1.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

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
    stability_weights = [0.35, 0.05, 0.05, 0.15, 0.3]
    z_scores['Stability'] = z_scores[stability_vars].dot(stability_weights)

    #add cushioning and stability to data columns
    data['Cushioning'] = z_scores['Cushioning']
    data['Stability'] = z_scores['Stability']
    
    return z_scores

def spiderplot_streamlit(z_scores, variables, shoe_names, labels, new_shoe_scores, plot_new_shoe):
    if not shoe_names and (not plot_new_shoe or new_shoe_scores is None):
        st.write("No shoes selected for plotting.")
        return

    angles = [n / float(len(variables)) * 2 * pi for n in range(len(variables))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], labels)

    for shoe_name in shoe_names:
        shoe_z_scores = z_scores[z_scores['Shoe Name'] == shoe_name]
        if not shoe_z_scores.empty:
            values = shoe_z_scores[variables].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=shoe_name)
            ax.fill(angles, values, alpha=0.1)

    # Plot the ideal shoe (user makes it up)
    if plot_new_shoe and new_shoe_scores is not None:
        new_shoe_values = np.append(new_shoe_scores, new_shoe_scores[0])
        ax.plot(angles, new_shoe_values, linewidth=2, linestyle='dotted', label='New Shoe', color='red')
        ax.fill(angles, new_shoe_values, alpha=0.2, color='red')

    
    ax.set_ylim(-2.5, 2)
    ax.set_yticklabels([])
    plt.title('Spider Plot')
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
    st.pyplot(fig)

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
    st.title("Shoe Spider Plot Generator")
    st.write("Select shoes to plot:")

    file_path = 'C:/Users/laura.healey/OneDrive - PUMA/Shared Inno Research/1. Running/2024June_SiloTestStrategy/Silo_testing_allshoes.xlsx'
    data = read_files(file_path)
    z_scores = get_z_scores(data)

    variables = ['Cushioning', 'Weight (g)', 'Rear_Energy Returned %', 'Stiffness (N.m/deg)', 'Stability']
    labels = ['Cushioning', 'Lightweight', 'Energy Return', 'Bending Stiffness', 'Stability']
  
    # Checkbox to determine if new shoe should be plotted
    plot_new_shoe = st.sidebar.checkbox("Include New Shoe in Plot", value=True)
    
    # Define custom slider ranges for each variable
    slider_ranges = {
        'Cushioning': {'min': 0.5, 'max': 2.0, 'step': 0.05},
        'Weight (g)': {'min': 50.0, 'max': 350.0, 'step': 5.0},
        'Rear_Energy Returned %': {'min': 60.0, 'max': 100.0, 'step': 0.5},
        'Stiffness (N.m/deg)': {'min': 0.1, 'max': 0.6, 'step': 0.1},
        'Stability': {'min': -0.5, 'max': 2.0, 'step': 0.01}
    }
    
    #plot a new shoe
    means = data[variables].mean()
    std_devs = data[variables].std()

    # Initialize new_shoe_scores
    new_shoe_scores = None  # Default to None
    
    if plot_new_shoe:
        st.sidebar.header("New Shoe Characteristics")

        new_shoe_scores_raw = []  # Store raw scores
        new_shoe_scores = []    # Store z-scores
        for var in variables:
            # Get custom range values for this variable
            min_val = slider_ranges[var]['min']
            max_val = slider_ranges[var]['max']
            step_val = slider_ranges[var]['step']
           
            # Slider for raw number input
            raw_score = st.sidebar.slider(
                f"{var} for New Shoe",
                min_value=min_val,
                max_value=max_val,
                value=means[var],  # Default to the mean of the variable
                step=step_val
            )
            new_shoe_scores_raw.append(raw_score)
            # Calculate z-score
            score = (raw_score - means[var]) / std_devs[var]

            # Calculate z-score and invert weight
            if var == 'Weight (g)':
                score = -(raw_score - means[var]) / std_devs[var]  # Invert weight
            else:
                score = (raw_score - means[var]) / std_devs[var]

            new_shoe_scores.append(score)

    else:
        new_shoe_scores_raw = None
        new_shoe_scores = None


    
    # if plot_new_shoe:
    #     st.sidebar.header("New Shoe Characteristics")
    #     new_shoe_scores = []
    #     for var in variables:
    #         score = st.sidebar.slider(f"{var} for New Shoe", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
    #         new_shoe_scores.append(score)
    # else:
    #     new_shoe_scores = None

    # Multi-select for existing shoes in three columns with checkboxes
    shoe_names = z_scores['Shoe Name'].tolist()
    num_columns = 3
    cols = st.columns(num_columns)
    
    selected_shoes = []
    for i, col in enumerate(cols):
        with col:
            start_idx = i * len(shoe_names) // num_columns
            end_idx = (i + 1) * len(shoe_names) // num_columns
            for shoe_name in shoe_names[start_idx:end_idx]:
                if st.checkbox(shoe_name, key=shoe_name):
                    selected_shoes.append(shoe_name)

    # Plot the spider graph live as the sliders or checkboxes change
    spiderplot_streamlit(z_scores, variables, selected_shoes, labels, new_shoe_scores, plot_new_shoe)

    
    #COMPARISON TOOL
    # Comparison Tool Section
    st.header("Shoe Comparison Tool")

    if st.button("Compare Shoes"):
        if len(selected_shoes) < 2:
            st.error("Please select at least two shoes to compare.")
        else:
            most_similar_pair, distance_df = compare_shoes(z_scores, selected_shoes, variables)
                
            flat_distances = (
                distance_df.where(np.triu(np.ones(distance_df.shape), k=1).astype(bool))  # Only upper triangle (excluding diagonal)
                .stack()
            )

            # Sort the flattened distances and select the top 3
            top3_similar_pairs = flat_distances.nsmallest(3)

            # Display the top 3 most similar pairs of shoes
            st.write("Top 3 most similar pairs of shoes:")
            for idx, (shoes, distance) in enumerate(top3_similar_pairs.items(), 1):
                st.write(f"#{idx} Most similar pair: {shoes} with a distance of {distance:.2f}")
                        

            # Plotting the pairwise distance matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(distance_df, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
            plt.title("Pairwise Distance Matrix (Euclidean)")
            st.pyplot(plt)


if __name__ == "__main__":
    app()