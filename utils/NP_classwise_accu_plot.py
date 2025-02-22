import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_result_dict_by_sorted_names(result, save_path=None):
    # result is {name: accuracy}
    accuracy_dict = sorted(result.items(), key=lambda x: x[1], reverse=True)
    names, accus = zip(*accuracy_dict)
    # Plot using Matplotlib only
    plt.figure(figsize=(12, 16))
    plt.barh(names, accus, color='blue', alpha=0.7)  # Alpha for transparency

    plt.xlabel("Accuracy")
    plt.ylabel("Class")
    plt.title("Accuracy per Class")
    plt.xlim(0, 1.05)  # Adjust X-axis range
    plt.gca().invert_yaxis()  # Invert to match the original order
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        # You also needa save pkl here!
        pkl_path = save_path.replace('.png', '.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(result, f)
    else:
        plt.show()
        plt.close()
    
def ___compare_results_h(name1, result1, name2, result2, save_path=None):
    # result is {name: accuracy}
    accuracy_dict1 = sorted(result1.items(), key=lambda x: x[1], reverse=True)
    
    names, accus1 = zip(*accuracy_dict1)
    accus2 = [result2[name] for name in names]
    x = np.arange(len(names), dtype=float)
    bar_height = 0.15
    # Plot using Matplotlib only
    plt.figure(figsize=(12, 24))
    plt.barh(x-bar_height, accus1, height= 2*bar_height, color='blue', alpha=0.7, label=name1)  # Alpha for transparency
    plt.barh(x+bar_height, accus2, height= 2*bar_height, color='red', alpha=0.7, label=name2)  # Alpha for transparency

    plt.xlabel("Accuracy")
    plt.ylabel("Class")
    plt.title("Accuracy per Class")
    plt.xlim(0, 1.05)  # Adjust X-axis range
    plt.yticks(x, names)
    plt.gca().invert_yaxis()  # Invert to match the original order
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def compare_results(*name_and_accu_results, save_path=None, filter_by=None):
    # result is {name: accuracy}
    name1, result1 = name_and_accu_results[0]
    if filter_by is not None:
        result1 = {k:v for k,v in result1.items() if k in filter_by}
    accuracy_dict1 = sorted(result1.items(), key=lambda x: x[1], reverse=True)
    
    np_names, accus1 = zip(*accuracy_dict1)
    x = np.arange(len(result1), dtype=float)
    bar_width = 0.1
    x -= bar_width * (len(name_and_accu_results) - 1) / 2
    x_copy = x.copy()
    plt.figure(figsize=(20*len(name_and_accu_results), 24))
    for label_name, result in name_and_accu_results:
        
        accus = [result[name] for name in np_names]
        # Plot using Matplotlib only
        plt.bar(x, accus, width=bar_width, alpha=0.7, label=label_name)  # Alpha for transparency
        x += bar_width

    plt.ylabel("Accuracy", fontsize=48)
    plt.xlabel("Class", fontsize=48)
    plt.title("Accuracy per Class", fontsize=48)
    plt.ylim(0.3, 1.05)  # Adjust X-axis range
    plt.yticks(fontsize=48)
    plt.xticks(x_copy, np_names, rotation=30, fontsize=48)
    plt.legend(fontsize=48, bbox_to_anchor=(1.01, 1))
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()
    
    
# def compare_results_from_pkl_files(name1, pkl1, name2, pkl2, save_path=None):
def compare_results_from_pkl_files(*name_and_pkls, save_path=None, filter_by=None):
    name_and_accu_results = []
    for name_and_pkl in name_and_pkls:
        name, pkl = name_and_pkl
        with open(pkl, 'rb') as f:
            result = pickle.load(f)
        name_and_accu_results.append((name, result))
    # compare_results(result1, result2, save_path=save_path)
    compare_results(*name_and_accu_results, save_path=save_path, filter_by=filter_by)
    
    
