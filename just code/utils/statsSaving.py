import numpy as np
import time
import os
import pandas as pd
import pickle as pkl
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


class StatsSaving(object):
    """
    Class for statistics and
    saving the results
    """

    def __init__(self):
        # Initiation of metric scorer
        self.perfm_acc_tr = []
        self.perfm_acc_vld = []

        self.perfm_f1_tr = []
        self.perfm_f1_vld = []
        self.perfm_precision_tr = []
        self.perfm_precision_vld = []
        self.perfm_recall_tr = []
        self.perfm_recall_vld = []

        self.perfm_f1_tr_weighted = []
        self.perfm_f1_vld_weighted = []
        self.perfm_precision_tr_weighted = []
        self.perfm_precision_vld_weighted = []
        self.perfm_recall_tr_weighted = []
        self.perfm_recall_vld_weighted = []

        # self.perfm_roc_auc_score_tr = []
        # self.perfm_roc_auc_score_vld = []

        # self.perfm_TN_tr = []
        # self.perfm_FP_tr = []
        # self.perfm_FN_tr = []
        # self.perfm_TP_tr = []
        # self.perfm_TN_vld = []
        # self.perfm_FP_vld = []
        # self.perfm_FN_vld = []
        # self.perfm_TP_vld = []

        self.perfm_sensitivity_tr = []
        self.perfm_sensitivity_vld = []
        self.perfm_specificity_tr = []
        self.perfm_specificity_vld = []

        self.perfm_AUC_tr = []
        self.perfm_AUC_vld = []

        self.perfm_FPR_tr = np.array([])
        self.perfm_TPR_tr = np.array([])
        self.perfm_THRES_tr = np.array([])

        self.perfm_FPR_vld = np.array([])
        self.perfm_TPR_vld = np.array([])
        self.perfm_THRES_vld = np.array([])

        self.perfm_params_ = []
        self.perfm_boruta_features_ = []
        self.perfm_feature_importance_per_rf = []

    def results_and_stats(
        self,
        model,
        X,
        y,
        features,
        best_params,
    ):

        # TODO: temp solution for now
        X_test, y_test = X, y

        # Saving Boruta selected features
        self.perfm_boruta_features_.append(features)

        # Saving the best_params_ for this fold
        self.perfm_params_.append(best_params)

        self.perfm_feature_importance_per_rf.append(model.feature_importances_)

        self.perfm_predictions_tr = model.predict(X)
        self.perfm_predictions_tr_proba = model.predict_proba(X)

        # Predicting on the test fold
        self.perfm_predictions_vld = model.predict(X_test)
        self.perfm_predictions_vld_proba = model.predict_proba(X_test)

        # tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(
        #     y, self.perfm_predictions_tr
        # ).ravel()
        # tn_vld, fp_vld, fn_vld, tp_vld = confusion_matrix(
        #     y_test, self.perfm_predictions_vld
        # ).ravel()

        fpr_tr, tpr_tr, thresholds_tr = roc_curve(
            y,
            self.perfm_predictions_tr_proba[:, 1],
            pos_label=1,
            drop_intermediate=False,
        )
        fpr_vld, tpr_vld, thresholds_vld = roc_curve(
            y_test,
            self.perfm_predictions_vld_proba[:, 1],
            pos_label=1,
            drop_intermediate=False,
        )

        self.perfm_acc_tr.append(accuracy_score(y, self.perfm_predictions_tr))

        # Uncomment for debugging
        # print(f"len: {len(perfm_acc_tr)}")

        self.perfm_acc_vld.append(accuracy_score(y_test, self.perfm_predictions_vld))

        self.perfm_f1_tr.append(f1_score(y, self.perfm_predictions_tr, average="macro"))
        self.perfm_f1_vld.append(
            f1_score(y_test, self.perfm_predictions_vld, average="macro")
        )
        self.perfm_precision_tr.append(
            precision_score(y, self.perfm_predictions_tr, average="macro")
        )
        self.perfm_precision_vld.append(
            precision_score(y_test, self.perfm_predictions_vld, average="macro")
        )
        self.perfm_recall_tr.append(
            recall_score(y, self.perfm_predictions_tr, average="macro")
        )
        self.perfm_recall_vld.append(
            recall_score(y_test, self.perfm_predictions_vld, average="macro")
        )

        self.perfm_f1_tr_weighted.append(
            f1_score(y, self.perfm_predictions_tr, average="weighted")
        )
        self.perfm_f1_vld_weighted.append(
            f1_score(y_test, self.perfm_predictions_vld, average="weighted")
        )
        self.perfm_precision_tr_weighted.append(
            precision_score(y, self.perfm_predictions_tr, average="weighted")
        )
        self.perfm_precision_vld_weighted.append(
            precision_score(y_test, self.perfm_predictions_vld, average="weighted")
        )
        self.perfm_recall_tr_weighted.append(
            recall_score(y, self.perfm_predictions_tr, average="weighted")
        )
        self.perfm_recall_vld_weighted.append(
            recall_score(y_test, self.perfm_predictions_vld, average="weighted")
        )

        # self.perfm_roc_auc_score_tr.append(
        #     roc_auc_score(y, self.perfm_predictions_tr_proba[:, 1])
        # )
        # self.perfm_roc_auc_score_vld.append(
        #     roc_auc_score(y_test, self.perfm_predictions_vld_proba[:, 1])
        # )

        self.perfm_AUC_tr.append(auc(fpr_tr, tpr_tr))
        self.perfm_AUC_vld.append(auc(fpr_vld, tpr_vld))

        # self.perfm_TN_tr.append(tn_tr)
        # self.perfm_FP_tr.append(fp_tr)
        # self.perfm_FN_tr.append(fn_tr)
        # self.perfm_TP_tr.append(tp_tr)
        # self.perfm_TN_vld.append(tn_vld)
        # self.perfm_FP_vld.append(fp_vld)
        # self.perfm_FN_vld.append(fn_vld)
        # self.perfm_TP_vld.append(tp_vld)

        self.perfm_FPR_tr = np.append(self.perfm_FPR_tr, fpr_tr)
        self.perfm_TPR_tr = np.append(self.perfm_TPR_tr, tpr_tr)
        self.perfm_THRES_tr = np.append(self.perfm_THRES_tr, thresholds_tr)

        self.perfm_FPR_vld = np.append(self.perfm_FPR_vld, fpr_vld)
        self.perfm_TPR_vld = np.append(self.perfm_TPR_vld, tpr_vld)
        self.perfm_THRES_vld = np.append(self.perfm_THRES_vld, thresholds_vld)

        # self.perfm_sensitivity_tr.append(
        #     self.perfm_TP_tr[-1] / (self.perfm_TP_tr[-1] + self.perfm_FN_tr[-1])
        # )
        # self.perfm_sensitivity_vld.append(
        #     self.perfm_TP_vld[-1] / (self.perfm_TP_vld[-1] + self.perfm_FN_vld[-1])
        # )
        # self.perfm_specificity_tr.append(
        #     self.perfm_TN_tr[-1] / (self.perfm_TN_tr[-1] + self.perfm_FP_tr[-1])
        # )
        # self.perfm_specificity_vld.append(
        #     self.perfm_TN_vld[-1] / (self.perfm_TN_vld[-1] + self.perfm_FP_vld[-1])
        # )


def LogSave(obj, ct, logger, name_experiment, PATH, start_time, *args, **kwargs):
    """
    Function that saves and logs results
    incrementally and totally
    """
    # Saving the incremental results on the output file
    if ct.stats_inc_output and args:

        # Writing everything to the logger
        logger.info(
            "\n--- Incremental Evaluation metrics, EGO with "
            + str(ct.ho_max_step)
            + " steps."
            + " ---\n"
        )
        logger.info("acc train score " + ": " + str(obj.perfm_acc_tr[-1]) + "\n")
        logger.info(
            "acc optimization_measure_mean score "
            + ": "
            + str(obj.perfm_acc_vld[-1])
            + "\n"
        )
        logger.info("f1 train score (macro) " + ": " + str(obj.perfm_f1_tr[-1]) + "\n")
        logger.info(
            "f1 optimization_measure_mean score (macro) "
            + ": "
            + str(obj.perfm_f1_vld[-1])
            + "\n"
        )

        lst_files = []
        keys = []
        values = []
        keys.append("n_iteration")
        values.append(args[0])
        keys.append("run_id")
        values.append(name_experiment)
        keys.append("date_time")
        values.append(name_experiment[:15])
        keys.append("dataset")
        values.append(name_experiment[16:])
        keys.append("computation_time")
        values.append(time.time() - start_time)

        for measure in [a for a in obj.__dict__.keys() if a.startswith("perfm_")]:
            if(len(eval("obj." + measure))>0): 
                keys.append(str(measure))
                values.append(eval("obj." + measure)[-1])

        zipped = zip(keys, values)
        a_dictionary = dict(zipped)
        lst_files.append(a_dictionary)
        df_performance_inc = pd.DataFrame.from_dict(lst_files)

        try:
            df_performance_inc = df_performance_inc.append(
                pd.read_csv(
                    os.path.join(
                        PATH,
                        name_experiment,
                        name_experiment + "_performance_inc.csv",
                    ),
                    sep=";",
                )
            )
        except:
            logger.info(
                "Appending df_performance_inc has failed, first iteration? " + "\n"
            )

        df_performance_inc.to_csv(
            os.path.join(
                PATH,
                name_experiment,
                name_experiment + "_performance_inc.csv",
            ),
            index=False,
            sep=";",
        )

    if kwargs:
        lst_files = []
        keys = []
        values = []
        # keys.append("n_iteration")
        # values.append(i)
        keys.append("run_id")
        values.append(name_experiment)
        keys.append("date_time")
        values.append(name_experiment[:15])
        keys.append("dataset")
        values.append(name_experiment[16:])
        keys.append("computation_time")
        values.append(time.time() - start_time)

        for measure in [a for a in obj.__dict__.keys() if a.startswith("perfm_")]:
            keys.append(str(measure))
            try:
                values.append(np.mean(eval("obj." + measure)))
            except:
                values.append(eval("obj." + measure))

        zipped = zip(keys, values)
        a_dictionary = dict(zipped)
        lst_files.append(a_dictionary)
        df_performance_total = pd.DataFrame.from_dict(lst_files)

        try:
            df_performance_total = df_performance_total.append(
                pd.read_csv(
                    os.path.join(
                        PATH,
                        name_experiment,
                        name_experiment + "_performance_total.csv",
                    ),
                    sep=";",
                )
            )
        except:
            logger.info(
                "Appending df_performance_total has failed, first iteration? " + "\n"
            )

        df_performance_total.to_csv(
            os.path.join(
                PATH,
                name_experiment,
                name_experiment + "_performance_total.csv",
            ),
            index=False,
            sep=";",
        )

        # Saving all the lists and metrics
        # saving the parameters per split:
        for measure in [a for a in dir() if a.startswith("self.perfm_")]:
            with open(
                os.path.join(PATH, name_experiment, "pkls", measure + ".pkl"),
                "wb",
            ) as f:
                pkl.dump(eval(measure), f)

        logger.info(
            "\n--- run time: " + str((time.time() - start_time) / 60) + " min ---\n"
        )
        logger.info("\n--- D O N E ---\n")
