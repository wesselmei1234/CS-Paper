import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from func import bootstrap, run_iteration
from tv import create_tv_list

# import data
JSON_FILE_PATH = 'TVs-all-merged.json'
with open(JSON_FILE_PATH, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# create a list of all tvs
tv_data = create_tv_list(data)

# create train and test set
tv_list, tv_list_test = bootstrap(tv_data)

# run bootstrap
t_options = np.arange(0.1, 1.1, 0.1)
number_of_hashes = 840
iterations = 5
performance_measures = {}

for t in t_options:
    final_results = np.zeros(10)
    for iteration in range(iterations):
        results = run_iteration(tv_list, tv_list_test, t, number_of_hashes)
        final_results = [x + y for x, y in zip(results, final_results)]

    final_results = [value / iterations for value in final_results]
    performance_measures[t] = final_results


file_path = '/Users/wesselmeiring/Documents/performance.json'
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(performance_measures, file)


# Extract performance measures 
fraction_comparisons = []
f1 = []
precision = []
recall = []
f1star = []
PC = []
PQ = []


for measures in performance_measures.values():
    fraction_comparisons.append(measures[6])
    f1.append(measures[0])
    precision.append(measures[1])
    recall.append(measures[2])
    f1star.append(measures[3])
    PC.append(measures[4])
    PQ.append(measures[5])

# Sort the data based on fraction comparisons
zipped_lists = list(zip(fraction_comparisons, f1, precision, recall, f1star, PC, PQ))
corresponding_elements = sorted(zipped_lists, key=lambda x: x[0])
fraction_comparisons, f1, precision, recall, f1star, PC, PQ = zip(*corresponding_elements)

# Set Seaborn style
sns.set(style="whitegrid")

# Create a single plot for F1, Precision, and Recall
plt.figure(figsize=(12, 6))
sns.lineplot(x=fraction_comparisons, y=f1, label='F1 Score', color='blue')
sns.lineplot(x=fraction_comparisons, y=precision, label='Precision', color='red')
sns.lineplot(x=fraction_comparisons, y=recall, label='Recall', color='green')

plt.title('F1, Precision, and Recall vs Fraction Comparisons', fontsize=16)
plt.xlabel('Fraction Comparisons', fontsize=14)
plt.ylabel('Performance Measures', fontsize=14)
plt.legend(fontsize=12)
plt.savefig('f1_precision_recall_plot.png') 
plt.close() 

# Create individual plots for F1star, PC, and PQ
performance_measures = [f1star, PC, PQ]
measure_names = ['F1 Star', 'PC', 'PQ']

for measure, measure_name in zip(performance_measures, measure_names):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=fraction_comparisons, y=measure, label=measure_name, color='red')
    plt.title(f'{measure_name} vs Fraction Comparisons', fontsize=16)
    plt.xlabel('Fraction Comparisons', fontsize=14)
    plt.ylabel(measure_name, fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig(f'{measure_name.lower()}_plot.png') 
    plt.close()  
