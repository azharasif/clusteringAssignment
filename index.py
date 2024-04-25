"""
Created on Thu Apr 23 06:57:20 2024

@author: azhar
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score

# Load the dataset from a CSV file
def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

# Preprocess data: convert 'year' and 'month' columns to integers, create 'date' column
def preprocess_data(data):
    """Preprocess data: convert 'year' and 'month' columns to integers, create 'date' column."""
    data['year'] = data['year'].astype(int)
    data['month'] = data['month'].astype(int)
    data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
    return data

# Aggregate data annually based on 'year'
def aggregate_annually(data):
    """Aggregate data annually based on 'year'."""
    return data.groupby('year')['value'].sum().reset_index()

# Plot annual trends
def plot_annual_trends(data):
    """Plot annual trends."""
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=data, x='year', y='value', marker='o', color='blue')
    plt.title('Annual Natural Gas Consumption', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Total Consumption (MMcf)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# Calculate year-over-year change in natural gas consumption
def calculate_yoy_change(data):
    """Calculate year-over-year change in natural gas consumption."""
    data['YoY_change'] = data['value'].pct_change() * 100
    return data

# Plot year-over-year change in natural gas consumption
def plot_yoy_change(data):
    """Plot year-over-year change in natural gas consumption."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x='year', y='YoY_change', palette='coolwarm')
    plt.title('Change in Natural Gas Consumption from 2014 to 2024', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('YoY Change (%)', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.axhline(0, color='gray', lw=1, linestyle='--') 
    plt.show()

# Calculate inter-sector correlation of natural gas consumption
def calculate_inter_sector_correlation(data):
    """Calculate inter-sector correlation of natural gas consumption."""
    sector_df = data.groupby(['year', 'process-name'])['value'].sum().reset_index()
    pivot_sector_df = sector_df.pivot(index='year', columns='process-name', values='value')
    return pivot_sector_df.corr()

# Plot inter-sector correlation of natural gas consumption
def plot_inter_sector_correlation(correlation_matrix):
    """Plot inter-sector correlation of natural gas consumption."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Natural Gas Consumption within Different sectors', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()

# Plot elbow plot for K-means clustering
def plot_elbow_plot(X, silhouette_score):
    """Plot elbow plot for K-means clustering."""
    plt.figure(figsize=(12, 6))
    inertia = []
    for n in range(1, 11):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.ylabel('Inertia', fontsize=14)
    plt.title('Elbow Plot for K-means Clustering', fontsize=16)
    plt.annotate('Elbow Point (3 clusters)', xy=(3, inertia[2]), xytext=(3, inertia[2] + 1000),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)
    plt.text(0.5, 0.95, f'Silhouette Score: {silhouette_score}', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# Perform K-means clustering
def perform_kmeans_clustering(X):
    """Perform K-means clustering."""
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    return kmeans, kmeans.labels_

# Plot K-means clusters
def plot_kmeans_clusters(X, labels, cluster_centers):
    """Plot K-means clusters."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=X, x='year', y='value', hue=labels, palette='viridis', legend='full')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.title('K-means Clustering of Annual Natural Gas Consumption', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Total Consumption (MMcf)', fontsize=14)
    plt.legend(title='Cluster', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.show()

# Fit a linear line
def fit_linear_line(data):
    """Fit a linear line."""
    plt.figure(figsize=(12, 6))
    plt.scatter(data['year'], data['value'], color='blue', label='Data')
    z = np.polyfit(data['year'], data['value'], 1)
    p = np.poly1d(z)
    data['linear_fit'] = p(data['year'])
    return p, data

# Fitting Prediction for year 2025 and 2026
def fitting_prediction(p):
    """Fitting Prediction for year 2025 and 2026."""
    new_x = np.array([2025, 2026])
    new_y_predicted = p(new_x)
    return new_x, new_y_predicted

# Visualize the fitting predictions
def visualize_fitting_predictions(data, p, new_x, new_y_predicted):
    """Visualize the fitting predictions."""
    plt.plot(data['year'], data['linear_fit'], color='red', label='Linear Fit')
    plt.scatter(new_x, new_y_predicted, color='green', label='Fitting Predictions')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Total Consumption (MMcf)', fontsize=14)
    plt.title('Fitting Prediction of Annual Natural Gas Consumption', fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.show()

def main():
    # Load data
    data = load_data('./data.csv')

    # Preprocess data
    data = preprocess_data(data)

    # Aggregate data annually
    annual_df = aggregate_annually(data)

    # Plot annual trends
    plot_annual_trends(annual_df)

    # Calculate year-over-year change
    annual_df = calculate_yoy_change(annual_df)

    # Plot year-over-year change
    plot_yoy_change(annual_df)

    # Calculate inter-sector correlation
    correlation_matrix = calculate_inter_sector_correlation(data)

    # Plot inter-sector correlation
    plot_inter_sector_correlation(correlation_matrix)

    # Perform K-means clustering
    kmeans, labels = perform_kmeans_clustering(annual_df[['year', 'value']])

    # Plot K-means clusters
    plot_kmeans_clusters(annual_df[['year', 'value']], labels, kmeans.cluster_centers_)

    # Fit a linear line
    p, annual_df = fit_linear_line(annual_df)

    # Fitting Prediction
    new_x, new_y_predicted = fitting_prediction(p)
    visualize_fitting_predictions(annual_df, p, new_x, new_y_predicted)

    # Calculate silhouette score for K-means clustering
    silhouette_avg = silhouette_score(annual_df[['year', 'value']], labels)
    print("Silhouette Score:", silhouette_avg) 

    # Plot elbow plot for K-means clustering
    plot_elbow_plot(annual_df[['year', 'value']], silhouette_avg)

if __name__ == "__main__":
    main()
