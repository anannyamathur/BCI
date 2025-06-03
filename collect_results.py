import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import numpy as np

# File paths
folder= "Results_Folder/"

your_model_file = folder +"parameters.csv"
sota_file = folder + "SOTA/" + "parameters.csv"

# Directory to save the plots
output_dir = folder+ "plots"
os.makedirs(output_dir, exist_ok=True)

# Load the CSV files
your_model_data = pd.read_csv(your_model_file)
sota_data = pd.read_csv(sota_file)

# Parse the list columns (assuming they are stored as strings in the CSV)
your_model_data["accuracy"] = your_model_data["accuracy"].apply(ast.literal_eval)
sota_data["accuracy"] = sota_data["accuracy"].apply(ast.literal_eval)


# Compute mean and standard deviation directly from the lists
your_model_data["MeanAccuracy"] = your_model_data["accuracy"].apply(lambda x: sum(x) / len(x))
your_model_data["StdAccuracy"] = your_model_data["accuracy"].apply(lambda x: pd.Series(x).std())
your_model_data["Model"] = "Our Model"

sota_data["MeanAccuracy"] = sota_data["accuracy"].apply(lambda x: sum(x) / len(x))
sota_data["StdAccuracy"] = sota_data["accuracy"].apply(lambda x: pd.Series(x).std())
sota_data["Model"] = "SOTA"

# Combine the data
combined_data = pd.concat([your_model_data, sota_data], ignore_index=True)

# Set theme for plots
sns.set_theme(style="whitegrid")

# Line plot with error bars
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=combined_data,
    x="training steps",
    y="MeanAccuracy",
    hue="Model",
    style="Model",
    markers=True,
    errorbar=None
)

# Add error bars manually
for model in ["Our Model", "SOTA"]:
    model_data = combined_data[combined_data["Model"] == model]
    plt.errorbar(
        model_data["training steps"],
        model_data["MeanAccuracy"],
        yerr=model_data["StdAccuracy"],
        fmt='o', capsize=5, alpha=0.7, label=f"{model} Error"
    )

plt.title("Comparison of Accuracy (Our Model vs SOTA) with Error Bars")
plt.xlabel("Training Time Steps")
plt.ylabel("Accuracy")
plt.legend(title="Model")
plt.grid(True)

plt.savefig(os.path.join(output_dir, "line_plot_error_bars.png"))
plt.close()

# Bar plot comparing mean accuracy
plt.figure(figsize=(12, 6))
sns.barplot(
    data=combined_data,
    x="training steps",
    y="MeanAccuracy",
    hue="Model",
    palette="viridis"
)
plt.title("Mean Accuracy Comparison Across Training Time Steps")
plt.xlabel("Training Time Steps")
plt.ylabel("Mean Accuracy")
plt.legend(title="Model")
plt.grid(True)

plt.savefig(os.path.join(output_dir, "bar_plot_comparison.png"))
plt.close()

# Heatmap for Our Model
plt.figure(figsize=(12, 6))
pivot_your_model = your_model_data.pivot_table(
    index="training steps",
    columns="testing steps",
    values="MeanAccuracy"
)
sns.heatmap(pivot_your_model, annot=True, cmap="Blues", cbar_kws={"label": "Accuracy"})
plt.title("Heatmap of Mean Accuracy (Our Model)")
plt.xlabel("Testing Time Steps")
plt.ylabel("Training Time Steps")

plt.savefig(os.path.join(output_dir, "heatmap_your_model.png"))
plt.close()

# Heatmap for SOTA Model
plt.figure(figsize=(12, 6))
pivot_sota = sota_data.pivot_table(
    index="training steps",
    columns="testing steps",
    values="MeanAccuracy"
)
sns.heatmap(pivot_sota, annot=True, cmap="Reds", cbar_kws={"label": "Accuracy"})
plt.title("Heatmap of Mean Accuracy (SOTA Model)")
plt.xlabel("Testing Time Steps")
plt.ylabel("Training Time Steps")
plt.savefig(os.path.join(output_dir, "heatmap_sota_model.png"))
plt.close()

######################################################################################

# Load the combined CSV file
df = combined_data

# Group by 'Model' and 'testing_steps' to find the maximum accuracy for each testing step
max_accuracy = df.groupby(['Model', 'testing steps'])['MeanAccuracy'].max().reset_index()

# Pivot the data to get a side-by-side comparison for each testing step
pivot_df = max_accuracy.pivot(index='testing steps', columns='Model', values='MeanAccuracy')

# Get the unique testing steps
testing_steps = pivot_df.index

# Create a bar plot
bar_width = 0.35
index = np.arange(len(testing_steps))

fig, ax = plt.subplots(figsize=(12, 6))

# Bars for Our Model
bars1 = ax.bar(index, pivot_df['Our Model'], bar_width, label='Our Model', color='blue')

# Bars for SOTA Model
bars2 = ax.bar(index + bar_width, pivot_df['SOTA'], bar_width, label='SOTA Model', color='red')

# Labeling the plot
ax.set_xlabel('Testing Steps')
ax.set_ylabel('Maximum Accuracy')
ax.set_title('Comparison of Maximum Accuracy Across Testing Steps')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(testing_steps)
ax.legend()

# Adding the values on top of the bars
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

# Save the plot
output_folder = output_dir+"/"
output_filename = output_folder + "max_accuracy_comparison_combined.png"
plt.savefig(output_filename)
print(f"Bar plot saved as {output_filename}")

# Display the plot
plt.close()