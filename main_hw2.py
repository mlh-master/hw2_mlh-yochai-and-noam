
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import random
from clean_data2 import rm_nan as rm
from clean_data2 import nan2num_samp
from clean_data2 import sum_stat
from clean_data2 import norm_standard
from sklearn.model_selection import train_test_split
from clean_data2 import compare_test_train_stas
from clean_data2 import plot_feature_diagnosis
from clean_data2 import cv_kfold
from clean_data2 import pred_log
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from clean_data2 import calculate
from sklearn.ensemble import RandomForestClassifier
from clean_data2 import important_features
from clean_data2 import D_redaction



if __name__ == '__main__':
    file = Path.cwd().joinpath('HW2_data.csv')  # concatenates messed_CTG.xls to the current folder that should
    # be the extracted zip folder
    T1D_dataset = pd.read_csv(file).iloc[:, :]
    T1D_features = T1D_dataset[['Age', 'Gender', 'Increased Urination', 'Increased Thirst', 'Sudden Weight Loss', 'Weakness', 'Increased Hunger', 'Genital Thrush', 'Visual Blurring', 'Itching', 'Irritability', 'Delayed Healing',
                                'Partial Paresis', 'Muscle Stiffness', 'Hair Loss', 'Obesity', 'Family History']]
    Diagnosis = T1D_dataset[['Diagnosis']]

    random.seed(10)

    ################ Q2 #####################

    c_t1d = rm(T1D_features, ['Age', 'Family History'])

    clean_t1d = nan2num_samp(c_t1d)

    d_summary = sum_stat(clean_t1d, 'Age')

    norm_t1d = norm_standard(clean_t1d, selected_feat='Age', mode='MinMax', flag=False)

    ################## Q3 ####################

    #stratify mean that in test and the training gonna be the same percent of samples from each labal- Negative/Posiive. that way we ensure they have the same distribution.
    #because we split the normelizes data, fron now on we dont need to scale the data (because we are using scaled data already
    X_train, X_test, y_train, y_test = train_test_split(norm_t1d, np.ravel(Diagnosis), test_size=0.2, random_state=336546, stratify=np.ravel(Diagnosis))

    compare_test_train_stas(X_test, X_train, 'Age')

    ##using the non_normalized data in order to get correct plot (normalization wont mettar for the plot):
    plot_feature_diagnosis(T1D_dataset, Diagnosis, ['Age', 'Diagnosis'])

    ##checking the relation between the two labels (additional plots us required):
    X = norm_t1d.iloc[:, 1:]
    Y = Diagnosis.iloc[:, :]
    Y.value_counts().plot(kind="pie", colors=['steelblue', 'salmon'], autopct='%1.1f%%')
    plt.title("The relation between the two labels")
    plt.show()

    #plotting the distribution of the age
    sns.histplot(x=T1D_dataset['Age'], data=T1D_dataset, hue="Diagnosis", shrink=.8)
    plt.show()

    ################## Q4 ####################

    #we are writing this for the record, but we alredy made our data to hot vector, without diagnosis.
    #(later, when w'll need it we will make the diagnosis to hot vector too).
    one_hot_vector = rm(T1D_dataset, ['Age', 'Family History'])

    ################## Q5 ####################

    C = [0.1, 1, 3, 10, 25, 100]  # a list of  6 different values of regularization parameters to examine their effects
    K = 5  # number of folds
    mode = 'MinMax'  # mode of norm_standard function

    y_train_df = pd.DataFrame(y_train)
    y_train_int = y_train_df.replace({"Positive": 1.0, "Negative": 0.0})
    validation_dict = cv_kfold(X_train, np.ravel(y_train_int), C=C, penalty=['l1', 'l2'], K=K, mode=mode) #np.ravel return a contiguous flattened array
    y_pred_non = pred_log(SVC(C=validation_dict["non_linear"]["C"], kernel='poly', max_iter=10000), X_train, np.ravel(y_train_int), X_test)
    y_pred_lin = pred_log(LogisticRegression(solver='saga', penalty=validation_dict["linear"]["penalty"], C=validation_dict["linear"]["C"], max_iter=10000, multi_class='ovr'), X_train,np.ravel(y_train_int), X_test)

    y_test_df = pd.DataFrame(y_test)
    y_test_int = y_test_df.replace({"Positive": 1.0, "Negative": 0.0})

    calculate(np.ravel(y_test_int), y_pred_non, "svm - non linear model", validation_dict["non_linear"]["mu"])
    calculate(np.ravel(y_test_int), y_pred_lin, "logistic regression - linear model", validation_dict["linear"]["mu"])


    ####################Q6#######################

    clf = RandomForestClassifier(max_depth=2, random_state=10)
    clf.fit(X_train, y_train)
    print(clf.feature_importances_)

    feat1,feat2 = important_features(clf.feature_importances_,['Age', 'Gender', 'Increased Urination', 'Increased Thirst', 'Sudden Weight Loss', 'Weakness', 'Increased Hunger', 'Genital Thrush', 'Visual Blurring', 'Itching', 'Irritability', 'Delayed Healing',
                                'Partial Paresis', 'Muscle Stiffness', 'Hair Loss', 'Obesity', 'Family History'])
    print("The 2 most important features according to the random forest are:", feat1, "and", feat2)


    ####################Q7#######################

    hot_vec_diagnosis = Diagnosis.replace({"Positive": 1.0, "Negative": 0.0})
    #D_redaction(data=X_train, num_component =2, labels=hot_vec_diagnosis) #doesnt work from some reason

    #finding the best hyper-parameter c and penalty
    X_train_2feat = X_train[['Increased Urination', 'Increased Thirst']]
    X_test_2feat = X_test[['Increased Urination', 'Increased Thirst']]
    C = [0.1, 1, 3, 10, 25, 100]  # a list of  6 different values of regularization parameters to examine their effects
    K = 5  # number of folds
    mode = 'MinMax'  # mode of norm_standard function
    validation_dict = cv_kfold(X_train_2feat, np.ravel(y_train_int), C=C, penalty=['l1', 'l2'], K=K, mode=mode)

    y_pred_non = pred_log(SVC(C=validation_dict["non_linear"]["C"], kernel='poly', max_iter=10000), X_train_2feat, np.ravel(y_train_int), X_test_2feat)
    y_pred_lin = pred_log(LogisticRegression(solver='saga', penalty=validation_dict["linear"]["penalty"], C=validation_dict["linear"]["C"], max_iter=10000, multi_class='ovr'), X_train_2feat, np.ravel(y_train_int), X_test_2feat)








