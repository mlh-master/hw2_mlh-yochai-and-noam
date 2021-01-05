
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from IPython.display import display


def rm_nan(T1D_features,ignore_list):
    """
    :param T1D_features: Pandas series of T1D features
    :return: A dictionary of clean T1D called c_t1d
    """
    c_t1d = {}  # create a dictionary
    for i in T1D_features:
        if(i not in ignore_list):
            c_t1d[i] = T1D_features[i].replace({"Yes": 1.0, "No": 0.0, "Positive": 1.0, "Negative": 0.0, "Female": 1.0, "Male": 0.0, "": np.nan})
        else:
            c_t1d[i] = T1D_features[i]
    return pd.DataFrame(c_t1d)


def nan2num_samp(T1D_features):
    """
    :param T1D_features: Pandas series of TD1 features
    :return: A pandas dataframe of the dictionary c_t1d containing the "clean" features
    """
    clean_t1d = {}
    for i in T1D_features:
        clean_t1d[i] = pd.to_numeric(T1D_features[i], errors='coerce')  # in every coloumn- every string convert to numeric, and if you cant- convert to nan

        col_without_nan = pd.to_numeric(clean_t1d[i], errors='coerce')  # in every coloumn- every string convert to numeric, and if you cant-nan
        col_without_nan = col_without_nan.dropna()  # drop all nan
        col_without_nan = col_without_nan.values  # convert to value instead of indexes

        for idx, val in enumerate(clean_t1d[i]): # enumerate returns a couple- count from 0 (idx) and list (val)/ the list is the values in the col c_cdf[i] (its type is const)
            if np.isnan(val): # running on the list (val), anf if the value there is nan we are getting in
                rand_idx = np.random.choice(len(col_without_nan)) # choosing a random index from the col_without_nan
                clean_t1d[i][idx] = col_without_nan[rand_idx]

    clean_t1d = pd.DataFrame(clean_t1d)
    clean_t1d = clean_t1d.astype(float)
    return clean_t1d


def sum_stat(clean_t1d, feature):
    """
    :param clean_t1d: Output of nan2num
    :param feature: The column we'd like to know what its summary statistics
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    d_summary = {}
    d_summary["min"] = np.min(clean_t1d[feature])
    d_summary["Q1"] = np.quantile(a=clean_t1d[feature], q=0.25)
    d_summary["median"] = np.median(a=clean_t1d[feature])
    d_summary["Q3"] = np.quantile(a=clean_t1d[feature], q=0.75)
    d_summary["max"] = np.max(clean_t1d[feature])
    return d_summary

def norm_standard(data, selected_feat='Age', mode='none', flag=False):
    """
    :param data: Pandas series of T1D features
    :param selected_feat: An element tuple of string of the features we'd like to normalize
    :param mode: A string determining the mode - normalization type
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called
    """
    x = selected_feat
    norm_t1d = data
    minX = min(data[x])
    maxX = max(data[x])
    meanX = np.mean(data[x])
    if (mode == 'MinMax'):
        norm_t1d[selected_feat] = (data[x] - minX) / (maxX-minX)
    elif (mode == 'mean'):
        norm_t1d[selected_feat] = (data[x] - meanX) / (maxX-minX)
    elif (mode == 'standard'):
        norm_t1d[selected_feat] = (data[x] - meanX) / np.std(data[x])
    else:
        norm_t1d[selected_feat] = data[x]
    if (flag):
        Q_clean = pd.DataFrame(data[selected_feat])
        Q_clean.hist(bins=100)
        plt.xlabel('Histogram Width')
        plt.ylabel('Count')
        plt.show()
    return norm_t1d.astype(float)


def compare_test_train_stas(test,train,ecepional_feat):
    """"
    :param test: Output of train_test_split
    :param train: Output of train_test_split
    :param eceptional_feat: not boolian feature that will treated different
    :return: table that compare between statistics of each feature in train and in the test samples
    """
    pos_fet_arr = []
    test_arr = []
    train_arr = []
    delta_arr = []
    train_len = train.shape[0]
    test_len = test.shape[0]
    for feat in test:
        if feat != ecepional_feat:
            pos_fet_arr.append(feat)
            test_res = round(test[feat].sum()/test_len*100, 1)
            test_arr.append(test_res)
            train_res = round(train[feat].sum()/train_len*100, 1)
            train_arr.append(train_res)
            delta_arr.append(round(train_res - test_res, 1))
        else:
            continue
    table = pd.DataFrame({'positive feature': pos_fet_arr, 'test%': test_arr, 'train%': train_arr, 'delta%': delta_arr})
    return display(table)

def plot_feature_diagnosis(data, diagnosis, ecepional_features):
    """"
    :param data:the non_normalized data
    :param diagnosis: the column diagnosis
    :param eceptional_feat: not boolian feature that will treated different
    :return: Plots that shows the relationship between each feature and label.
    """
    for feature in data:
          if feature not in ecepional_features:
            g = sns.countplot(x=feature, data=data, hue="Diagnosis")
            plt.show()
          else:
            continue

def pred_log(model, X_train, y_train, X_test, linear=True):
    """
    :param model: An object of the class LogisticRegression
    :param X_train: Training set samples
    :param y_train: Training set labels
    :param X_test: Testing set samples
    :param linear: A boolean determining whether the model linear or non-linear
    :return: A two elements tuple containing the predictions and the weightning matrix
    """
    clf = model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    return y_pred_log


def cv_kfold(X, y, C, penalty, K, mode):
    """
    :param X: Training set samples
    :param y: Training set labels
    :param C: A list of regularization parameters
    :param penalty: A list of types of norm
    :param K: Number of folds
    :param mode: Mode of normalization (parameter of norm_standard function in clean_data module)
    :return: A dictionary which contain 2 features- linear (c,penalty and mean loss) and non-linear (c and mean loss).
    """
    kf = SKFold(n_splits=K)
    validation_dict = {}
    min_lin = math.inf
    min_non_lin = math.inf
    for c in C:
        for p in penalty:
            lin_model = LogisticRegression(solver='saga', penalty=p, C=c, max_iter=10000, multi_class='ovr')
            non_lin_model = SVC(C=c, kernel='poly', max_iter=10000)
            mu = run_model(lin_model, kf, X, y, mode, K)
            if (mu < min_lin):
                min_lin = mu
                validation_dict["linear"] = {'C': c, 'penalty': p, 'mu': mu}
            mu = run_model(non_lin_model, kf, X, y, mode, K)
            if (mu < min_non_lin):
                min_non_lin = mu
                validation_dict["non_linear"] = {'C': c,  'mu': mu}
    return validation_dict

def run_model(model,kf,X,y,mode,K):
    """
    :param model: linear or non linear
    :param kf: the k folds we made
    :param X: Training set samples
    :param y: Training set labels
    :param mode: Mode of normalization (parameter of norm_standard function in clean_data module)
    :param K: Number of folds
    :return: the mean loss of the k folds
    """
    loss_val_vec = np.zeros(K)
    k = 0
    for train_idx, val_idx in kf.split(X, y):
        x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        y_pred = pred_log(model, x_train, y_train, x_val)
        loss_val_vec[k] = log_loss(y_val, y_pred) # log_loss returns y_pred probabilities for its training data y_true
        k = k+1
    return loss_val_vec.mean()

calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

def calculate (y_test, y_pred_test, model_name, mean_loss):
    """
    :param y_test: testing set labels
    :param y_pred_test: the predicted labels to the test set due to our model
    :param model_name: linear or non linear
    :param K: Number of folds
    :return: print the AUC, ACC, F1 and LOSS
    """
    TN = calc_TN(y_test, y_pred_test)
    TP = calc_TP(y_test, y_pred_test)
    FN = calc_FN(y_test, y_pred_test)
    FP = calc_FP(y_test, y_pred_test)
    Se = TP / (TP + FN)
    Sp = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    F1 = (2 * PPV * Se) / (PPV + Se)
    AUC = roc_auc_score(y_test, y_pred_test)

    print("the evaluation metrics of the train and test sets of the", model_name, "is:", "AUC=",AUC, "F1=", F1, "ACC=", Acc, "LOSS:", mean_loss)

def important_features(feature_importances,feat_name):
    """
    :param feature_importances: array of feature importances - The values of this array sum to 1
    :return: 2 most important fetures
    """
    value1 = 0
    value2 = 0
    index1 = 0
    index2 = 0
    for idx, val in enumerate(feature_importances):
        if (val > value1):
            value2 = value1
            index2 = index1
            value1 = val
            index1 = idx
        elif ( val > value2):
            value2 = val
            index2 = idx
        else:
            continue
    return feat_name[index1],feat_name[ index2]

def D_redaction(data, num_component, labels):
    """
    :param data: the data we want to project in to a lower dimensional space
    :param num_component: Number of components to keep
    :param labels: diagnosis of the data
    :return: plot of the lower dimention data
    """
    pca =PCA(n_components=num_component, svd_solver='full', whiten=True)
    t_pca = pca.fit(data)
    plt_2d_pca(data=t_pca, labels=labels)
    #print(pca.singular_values_)

def plt_2d_pca(data,labels):
    """
    :param X_pca: the data we want to project in to a lower dimensional space, after fiting PCA
    :param labels: diagnosis of the data
    :return: plot of the lower dimention data
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(data[labels==0, 0], data[labels==0, 1], color='b')
    ax.scatter(data[labels==1, 0], data[labels==1, 1], color='r')
    ax.legend(('Positive','Negative'))
    ax.plot([0], [0], "ko")
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.set_xlabel('$U_1$')
    ax.set_ylabel('$U_2$')
    ax.set_title('2D PCA')