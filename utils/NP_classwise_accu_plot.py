import matplotlib.pyplot as plt

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
    else:
        plt.show()
        plt.close()
    
def compare_results(name1, result1, name2, result2, save_path=None):
    # result is {name: accuracy}
    accuracy_dict1 = sorted(result1.items(), key=lambda x: x[1], reverse=True)
    
    names, accus1 = zip(*accuracy_dict1)
    accus2 = [result2[name] for name in names]
    # Plot using Matplotlib only
    plt.figure(figsize=(12, 16))
    plt.barh(names, accus1, color='blue', alpha=0.5, label=name1)  # Alpha for transparency
    plt.barh(names, accus2, color='orange', alpha=0.5, label=name2)  # Alpha for transparency

    plt.xlabel("Accuracy")
    plt.ylabel("Class")
    plt.title("Accuracy per Class")
    plt.xlim(0, 1.05)  # Adjust X-axis range
    plt.gca().invert_yaxis()  # Invert to match the original order
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
        
    
    plt.show()
    plt.close()


def compare_results_from_pkl_files(name1, pkl1, name2, pkl2, save_path=None):
    import pickle
    with open(pkl1, 'rb') as f:
        result1 = pickle.load(f)
    with open(pkl2, 'rb') as f:
        result2 = pickle.load(f)
    # compare_results(result1, result2, save_path=save_path)
    compare_results(name1, result1, name2, result2, save_path=save_path)