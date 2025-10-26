import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_cluster_scatter(df, x_col, y_col, cluster_col):
    """
    Visualisasi hasil clustering dalam bentuk scatter plot 2D.
    """
    fig, ax = plt.subplots()
    for c in df[cluster_col].unique():
        subset = df[df[cluster_col] == c]
        ax.scatter(subset[x_col], subset[y_col], label=f"Cluster {c}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    st.pyplot(fig)

def show_cluster_bar(df, cluster_col):
    """
    Visualisasi jumlah anggota per cluster (bar chart).
    """
    cluster_counts = df[cluster_col].value_counts().sort_index()
    st.bar_chart(cluster_counts)
