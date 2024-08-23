import streamlit as st
from pages import shoe_spider_plot  # Import the new page
from pages import shoe_comparison 
from pages import distance_speed_plotter

# Dictionary mapping page names to functions
pages = {
    #"Home": home.app,  # Assume you have a home page
    "Shoe Spider Plot": shoe_spider_plot.app,  # Add the new page
    "Shoe Comparison" : shoe_comparison.app,
    "Distance Speed Comparison" : distance_speed_plotter.app
}

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page
pages[page]()