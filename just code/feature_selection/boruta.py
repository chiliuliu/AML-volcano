from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def boruta_feature_selection(X, Y):  # X_test can be added here
    """
    Selecting promising features with boruta algorithm
    :param X: data features
    :param Y: data labels
    :return: feature vector, list of most selected features
    """

    rf = RandomForestClassifier(n_jobs=-1,
                                n_estimators=1000)

    feat_selector = BorutaPy(rf,
                             n_estimators=1000,
                             alpha=0.05,
                             max_iter=20, #here
                             verbose=2)

    feat_selector.fit(X, Y)
    # feat_selector.fit(X.values, Y)

    try:
        k = 0
        features = X.columns  # add here feature names
        features_list = []
        features_importance_list = []

        for i in feat_selector.support_:
            if i:
                features_list.append(features[k])
                features_importance_list.append(feat_selector.ranking_[k])
            k = k + 1

        features_list = [x for _, x in sorted(zip(features_importance_list, features_list))]
    except:
        features_list = ['no feature names given']

    # call transform() on X to filter it down to selected features
    X_boruta = feat_selector.transform(X)
    #X_boruta = feat_selector.transform(X.values)

    return X_boruta, features_list
