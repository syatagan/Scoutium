import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from Src.utils import *
from Src.conf import *
plt.matplotlib.use('Qt5Agg')

def read_scoutium_data():
    df_atts = pd.read_csv("Datasets/scoutium_attributes.csv",sep=";")
    df_labels = pd.read_csv("Datasets/scoutium_potential_labels.csv",sep=";")
    df= df_atts.merge(df_labels,on=["task_response_id", 'match_id', 'evaluator_id',"player_id"],how="inner")
    return df

def plot_importance(model, features, num=20, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')
def prepare_data(df):
    check_df(df)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    ## veri dengeli mi kontrol ediyorum.
    df["potential_label"].value_counts(normalize=True)
    ############################################
    # average          0.791892
    # highlighted      0.195433
    # below_average    0.012675
    df["potential_label"].hist()
    plt.show(block=True)
    ## stage 3 remove position_id=1 df[(df["position_id"] == 1)].shape 700 samples
    df = df[~(df["position_id"] == 1)]
    ## stage 4 remove potential_label =  below_average
    ## for test df[(df["potential_label"] == "below_average")].shape ## 136 samples
    df = df[~(df["potential_label"] == "below_average")]
    df.shape ## 9894 samples
    ################################################
    ## stage 5 create a pivot table
    pvt = pd.pivot_table(data=df,
                         index=["player_id","position_id","potential_label"],
                         columns=["attribute_id"],
                         values=["attribute_value"],
                         aggfunc="mean")

    pvt.head()
    pvt.reset_index(inplace=True)
    pvt.columns = pvt.columns.droplevel(0)
    cols = ["player_id","position_id","potential_label"]
    cols.extend(pvt.columns[3:].tolist())
    pvt.columns = cols
    pvt.columns = pvt.columns.astype(str)
    ################################################
    label_encoder(pvt , "potential_label")
    pvt["potential_label"].value_counts()

    num_cols = pvt.columns[3:].tolist()
    for col in num_cols:
        pvt[col] = StandardScaler().fit_transform(pvt[[col]])
    return pvt

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

def create_models(X, y, scoring = "roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier())
                   ]
    score_list = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    for model_name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=2, scoring=score_list)
        for score_name in score_list:
            print(f"{score_name}: {round(cv_results['test_'+ score_name].mean(), 4)} ({model_name}) ")

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    """
    classifiers = [('LR', LogisticRegression()),
                   ("RF", RandomForestClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier())
                   ]
    """
    cv = 2
    scoring = "f1"
    classifiers = [('LightGBM', LGBMClassifier(),{"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]})]
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('XGBoost', best_models["XGBoost"]), ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

def pipeline():
    df = read_scoutium_data()
    pvt_df = prepare_data(df)
    X = pvt_df.drop(["potential_label","player_id"],axis=1)
    y = pvt_df["potential_label"]
    create_models(X,y)
    best_models = hyperparameter_optimization(X,y,3)
    plot_importance()
    voting_classifier(best_models,X,y)

    cv = 2
    scoring = "f1"
    classifiers = [('LightGBM', LGBMClassifier(), {"learning_rate": [0.01, 0.1],
                                                   "n_estimators": [300, 500],
                                                   "colsample_bytree": [0.7, 1]})]
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
        best_models

        print("Voting Classifier...")
        voting_clf = VotingClassifier(estimators=[
                                                  ('LightGBM', best_models["LightGBM"])],
                                      voting='soft').fit(X, y)
        cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
        print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
        print(f"F1Score: {cv_results['test_f1'].mean()}")
        print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
        plot_importance(best_models['LightGBM'],len(X))