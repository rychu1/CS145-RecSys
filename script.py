import pandas as pd
import matplotlib.pyplot as plt
import io

# --- Data Loading ---
# The script will now attempt to load data from 'gcn_adaptive_search_results.csv'.
# Make sure this file is in the same directory as the script.
file_path = 'gcn_adaptive_search_results.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data from '{file_path}'")
except FileNotFoundError:
    print(f"Error: '{file_path}' not found.")
    print("Using fallback data to generate the plot.")
    # Fallback data if the file is not found
    csv_data = """embedding_size,num_layers,epochs,learning_rate,weight_decay,dropout,total_test_revenue
64,2,50,0.001,9e-05,0.2,25797.6184277234
64,2,50,0.001,9e-05,0.1,25729.464856659408
128,2,50,0.001,9e-05,0.1,25696.939124142915
64,2,50,0.001,1e-05,0.1,25624.189873085834
"""
    df = pd.read_csv(io.StringIO(csv_data))


# --- Visualization ---

# Check if the dataframe is empty
if df.empty:
    print("The DataFrame is empty. Cannot generate plot.")
else:
    # Columns to plot
    columns_to_plot = df.columns
    num_plots = len(columns_to_plot)

    # Create a figure and a set of subplots, one for each column
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(12, num_plots * 2.5), sharex=True)
    fig.suptitle('Hyperparameter and Revenue Values per Iteration', fontsize=16, y=0.99)

    # Iteration number will be the index of the DataFrame
    iterations = df.index

    # Create a subplot for each hyperparameter and for the revenue
    for i, col_name in enumerate(columns_to_plot):
        ax = axes[i]
        ax.plot(iterations, df[col_name], marker='o', linestyle='-')
        
        # Set titles and labels for each subplot
        ax.set_title(f'Trend for {col_name}')
        ax.set_ylabel(col_name)
        ax.grid(True)
        
        # Annotate each point with its value, formatted appropriately

    # Set the x-axis label only for the last subplot
    axes[-1].set_xlabel('Iteration Number')

    # Adjust layout to prevent titles from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Display the plot
    plt.show()

# --- Optional: Print DataFrame ---
# This part is just to show the data being used.
print("\n--- Data Used for Plotting ---")
print(df.head()) # Print the first 5 rows of the dataframe
