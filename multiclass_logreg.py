import pandas as pd
from OrdinalEncoder import OrdinalEncoder
from MinMaxScalerClass import MinMaxScaler
from OneVsAllClass import OneVsAll
from Utiles_Functions import print_confusion_matrix, load_and_scale, train_test_split
import matplotlib.pyplot as plt

"""
This function loads and scales the 'hsbdemo' dataset. It then performs a train-test split, 
initializes an instance of the OneVsAll class (which implements multiclass logistic regression 
using the one-vs-all approach), and fits the model on the training data.
"""
def run_hsbdemo():
    print("hsbdemo multiclass logistic regression")
    df, ordinals, scalers = load_and_scale('hsbdemo')

    X = df.drop(['prog'], axis=1).values
    Y = df['prog'].values

    x_test, x_train, y_test, y_train = train_test_split(0.2, X, Y)
    p = OneVsAll()
    p.fit(x_train, y_train)
    yhat = p.predict(x_test, threshold=0.5)
    M = p.confusion_matrix(x_test, y_test)
    p.print_confusion_matrix(M, ordinals['prog'])

    accuracy_by_class = p.accuracy_by_class(yhat, y_test)
    total_accuracy = p.accuracy(yhat, y_test)

    # Plot accuracy by class
    class_labels = list(accuracy_by_class.keys())
    class_accuracy = list(accuracy_by_class.values())

    plt.figure(figsize=(8, 6))
    plt.bar(class_labels, class_accuracy, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Class')

    # Add percentage values to the bars
    for i, v in enumerate(class_accuracy):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().yaxis.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Plot total accuracy
    plt.figure(figsize=(6, 4))
    plt.bar(['Total Accuracy'], [total_accuracy], color='skyblue')
    plt.ylim(0, 1)
    plt.xlabel('Metric')
    plt.ylabel('Accuracy')
    plt.title('Total Accuracy')

    # Add percentage value to the bar
    plt.text(0, total_accuracy + 0.01, f"{total_accuracy:.2f}", ha='center', va='bottom')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().yaxis.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print('accuracy by class values')
    for label, accuracy in accuracy_by_class.items():
        print(f'class = {label}, accuracy = {accuracy}')

    print(f'total accuracy = {p.accuracy(yhat, y_test)}')

###################################################################################
"""
This function loads the 'iris' dataset from the sklearn.datasets module. 
It creates a DataFrame from the dataset and performs ordinal encoding on the 'Target' column 
using the OrdinalEncoder class and min-max scaling on the remaining columns using the MinMaxScaler class. 
The function returns the preprocessed DataFrame, along with the ordinal encoders and scalers used.
"""
def load_iris_dataset():
    print("iris dataset multiclass logistic regression:")
    from sklearn import datasets
    data = datasets.load_iris()

    # create a DataFrame
    raw = pd.DataFrame(data.data, columns=data.feature_names)
    raw['Target'] = pd.DataFrame(data.target)
    ordinals = ['Target']

    # columns = ['ses', 'prog', 'write']
    # ordinals = ['ses', 'prog']

    ordinal_encoders = {}
    scalers = {}

    df = pd.DataFrame()
    for col in data.feature_names + ['Target']:
        if col in ordinals:
            encoder = OrdinalEncoder()
            df[col] = encoder.fit_transform(raw[col].values)
            ordinal_encoders[col] = encoder
        else:
            encoder = MinMaxScaler()
            df[col] = encoder.fit_transform(raw[col].values)
            scalers[col] = encoder
    return df, ordinal_encoders, scalers

"""
demonstrates multiclass logistic regression on the 'iris' dataset. 
It calls the load_iris_dataset function to obtain the preprocessed data,
ordinal encoders, and scalers. It then performs a train-test split, 
initializes an instance of the OneVsAll class, and fits the model on the training data
"""
def run_iris_dataset():
    df, ordinals, scalers = load_iris_dataset()

    X = df.drop(['Target'], axis=1).values
    Y = df['Target'].values

    x_test, x_train, y_test, y_train = train_test_split(0.2, X, Y)
    p = OneVsAll()
    p.fit(x_train, y_train, verbose=False, learning_rate=0.01, epochs=100000, print_every_n=1000)
    z = p.predict_probabilities(x_test)
    yhat = p.predict(x_test, threshold=0.5)
    M = p.confusion_matrix(x_test, y_test)
    p.print_confusion_matrix(M, ordinals['Target'])
    acc = p.accuracy_by_class(yhat, y_test)
    print('accuracy by class values')
    for k in acc:
        print(f'class = {k}, accuracy = {acc[k]}')

    print(f'total accuracy = {p.accuracy(yhat, y_test)}')

    total_accuracy = p.accuracy(yhat, y_test)

    # Plot the accuracies
    class_labels = list(acc.keys())
    accuracy_values = list(acc.values())

    plt.figure(figsize=(8, 6))
    plt.bar(class_labels, accuracy_values, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Class')

    # Add percentage values to the bars
    for i, v in enumerate(accuracy_values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().yaxis.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(['Total Accuracy'], [total_accuracy], color='skyblue')
    plt.ylim(0, 1)
    plt.xlabel('Metric')
    plt.ylabel('Accuracy')
    plt.title('Total Accuracy')

    # Add percentage value to the bar
    plt.text(0, total_accuracy + 0.01, f"{total_accuracy:.2f}", ha='center', va='bottom')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().yaxis.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_iris_dataset()
    print("##############################################################")
    run_hsbdemo()