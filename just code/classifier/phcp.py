import pickle as pkl
import time
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE,BorderlineSMOTE,ADASYN
import joblib
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    auc,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from utils.constants import Constants
from utils.statsSaving import StatsSaving, LogSave
from tqdm import tqdm

# from tsfresh.utilities.dataframe_functions import impute

from mipego import ParallelBO, RandomForest, NominalSpace, OrdinalSpace

from feature_selection.boruta import boruta_feature_selection


def phcp(PATH, name_experiment, logger, subpath, X, Y):
    """
    Plain hand-crafted pipeline, using boruta algorithm for feature selection, mip-ego for hpo
    :param PATH, str: basis path
    :param name_experiment, str: name of experiment
    :param logger: progress logger
    :param subpath: subpath indicating progress (e.g., baseline)
    :param X, np: data features
    :param Y, np: labels
    :return: None
    """
    ct = Constants(subpath)  # subpath object
    obj = StatsSaving()  # statistics and saving object

    # tracking time
    start_time = time.time()

    logger.info(
        "Parameters are: cv="
        + str(ct.cv)
        + ", random forest iterations="
        + str(ct.iterations_rf)
        + ", ho_max_step="
        + str(ct.ho_max_step)
        + ", optimize="
        + str(ct.to_optimize)
    )

    # only needed for imputing missing data, if exists
    #X = impute(X)

    logger.info("Started Boruta on the split")

    # X_train, features_list = boruta_feature_selection(X, Y) #X.values
    X_train=X
    features_list=['no feature names given']

    # Setting up the HO dataframe to save the results from all the HO iterations
    df_columns = [
        "performance_measure",
        "max_depth",
        "n_estimators",
        "bootstrap",
        "max_features",
        "min_samples_leaf",
        "min_samples_split",
    ]
    df_eval = pd.DataFrame(columns=df_columns)

    # Objective function for HO
    def obj_func(x):

        nonlocal df_eval
        nonlocal df_columns

        optimization_measure = []

        skf = StratifiedKFold(n_splits=ct.cv, random_state=np.random, shuffle=True)

        for train_index, test_index in skf.split(X, Y):
            X_train_BO, X_test_BO = X[train_index], X[test_index]
            y_train_BO, y_test_BO = Y[train_index], Y[test_index]

            smo = BorderlineSMOTE(random_state=42)
            X_train_BO, y_train_BO = smo.fit_resample(X_train_BO, y_train_BO)

            rf_opt = RandomForestClassifier(
                n_estimators=x["n_estimators"],
                max_depth=x["max_depth"],
                bootstrap=x["bootstrap"],
                max_features=x["max_features"],
                min_samples_leaf=x["min_samples_leaf"],
                min_samples_split=x["min_samples_split"],
                n_jobs=-1,
            )

            rf_opt.fit(X_train_BO, y_train_BO)

            predictions_ego_vld = rf_opt.predict(X_test_BO)

            # changed the score from acc to f1-macro!!!!!!! no, to weighted
            # also changed the accompanying df
            optimization_measure.append(
                f1_score(y_test_BO, predictions_ego_vld, average="weighted")
            )

        optimization_measure_mean = np.mean(optimization_measure)

        df_eval_tmp = pd.DataFrame(
            [
                [
                    optimization_measure_mean,
                    x["max_depth"],
                    x["n_estimators"],
                    x["bootstrap"],
                    x["max_features"],
                    x["min_samples_leaf"],
                    x["min_samples_split"],
                ]
            ],
            columns=df_columns,
        )
        df_eval = df_eval.append(df_eval_tmp)
        return optimization_measure_mean




    # Definition of hyperparameter search space:
    # max_depth = OrdinalSpace([2, 100], var_name='max_depth')  # NominalSpace([2, 100])
    max_depth = NominalSpace(
        [None] + np.arange(2, 102, 2).tolist(), var_name="max_depth"
    )  # NominalSpace([2, 100]) #originally should be 50, but NONE is added, so it turns to 51
    n_estimators = OrdinalSpace([1, 1000], var_name="n_estimators")
    min_samples_leaf = OrdinalSpace([1, 10], var_name="min_samples_leaf")
    min_samples_split = OrdinalSpace([2, 20], var_name="min_samples_split")
    bootstrap = NominalSpace(["True", "False"], var_name="bootstrap")
    max_features = NominalSpace(["auto", "sqrt", "log2"], var_name="max_features")

    search_space = (
        max_depth
        + n_estimators
        + bootstrap
        + max_features
        + min_samples_leaf
        + min_samples_split
    )
    model = RandomForest(levels=search_space.levels)

    opt = ParallelBO(
        search_space=search_space,
        obj_fun=obj_func,
        model=model,
        max_FEs=ct.ho_max_step,
        DoE_size=10,  # the initial DoE size
        eval_type="dict",
        acquisition_fun="MGFI",
        minimize=False,
        n_job=-1,  # number of processes
        n_point=3,  # number of the candidate solution proposed in each iteration
        verbose=False,  # turn this off, if you prefer no output
    )

    if ct.to_optimize:
        logger.info(
            "Started hyperparameter optimization using BO with ho_max_step: "
            + str(ct.ho_max_step)
        )
        opt.run() #first run to get the best result
        
    best_params_ = df_eval[df_columns[1:]][
        df_eval["performance_measure"] == df_eval["performance_measure"].max()
    ][:1].to_dict("records")

    # Saving the df_eval for every fold
    os.makedirs(os.path.join(PATH, name_experiment, "pkls"), exist_ok=False)
    with open(
        os.path.join(
            PATH,
            name_experiment,
            "pkls",
            "df_eval.pkl",
        ),
        "wb",
    ) as f:
        pkl.dump(df_eval, f)

    # Starting the training
    X=X_train
    for i in tqdm(range(0, ct.iterations_rf)):
    
    # for i in tqdm(range(0, 10)):
        logger.info("Counter is " + str(i))
        if ct.to_optimize:
            # Using the best_params froms the HO
            if(len(best_params_)>0):
                rf = RandomForestClassifier(n_jobs=-1, **best_params_[0])
            else:
                rf = RandomForestClassifier(n_jobs=-1)
        else:
            rf = RandomForestClassifier(n_jobs=-1)

        # TODO: add second ct.cv loop? remove/replace 3 lines below
    
        # Fitting the pipeline on the training fold
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10, random_state = 5)

        ada = BorderlineSMOTE(random_state=42)
        X_smo, y_smo = ada.fit_resample(X_train, y_train)

        rf.fit(X_smo, y_smo)

        obj.results_and_stats(
            rf,
            X_train,
            y_train,
            features_list,
            best_params_,
        )
        LogSave(obj, ct, logger, name_experiment, PATH, start_time, i)
        y_pred=rf.predict(X_test)
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print("F1 Score:",f1_score(y_test, y_pred, average='weighted'))
        logger.info("Accuracy:"+str(accuracy_score(y_test, y_pred)))
        logger.info("F1 Score:"+str(f1_score(y_test, y_pred, average='weighted')))

        if(i==0):
            joblib.dump(rf, 'rf.pkl') 

    LogSave(obj, ct, logger, name_experiment, PATH, start_time, final=True)
    print(best_params_[0])

    return (best_params_[0])
