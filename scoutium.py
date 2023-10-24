import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, roc_auc_score , accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from Src.utils import *
from Src.conf import *
import shap
plt.matplotlib.use('Qt5Agg')
warnings.simplefilter(action='ignore', category=FutureWarning)

model_metrics = pd.DataFrame(data=[])
def add_model_metric(xmodel, xauc, xaccuracy , xf1 ,xprecision, xrecall):
    new_row = {"Model" : xmodel, "Auc" : xauc , "Accuracy" : xaccuracy, "F1" : xf1 , "Precision" : xprecision , "Recall" : xrecall}
    return model_metrics.append(new_row,ignore_index=True)


df_atts = pd.read_csv("Datasets/scoutium_attributes.csv",sep=";")
df_labels = pd.read_csv("Datasets/scoutium_potential_labels.csv",sep=";")
df= df_atts.merge(df_labels,on=["task_response_id", 'match_id', 'evaluator_id',"player_id"],how="inner")

check_df(df)
cat_cols, num_cols, cat_but_car = grab_col_names(df)
## veri dengeli mi kontrol ediyorum.
print(df["potential_label"].value_counts(normalize=True))
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
print(pvt["potential_label"].value_counts())

num_cols = pvt.columns[3:].tolist()
for col in num_cols:
    pvt[col] = StandardScaler().fit_transform(pvt[[col]])

X = pvt.drop(["potential_label","player_id"],axis=1)
y = pvt["potential_label"]

print("Base Models....")
classifiers = [
               ("RF", RandomForestClassifier()),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
               ]
score_list = ["roc_auc", "f1", "precision", "recall", "accuracy"]
for model_name, classifier in classifiers:
    cv_results = cross_validate(classifier , X, y, cv=3, scoring=score_list)
    model_metrics = add_model_metric(model_name,
                     cv_results['test_roc_auc'].mean(),
                     cv_results['test_accuracy'].mean(),
                     cv_results['test_f1'].mean(),
                     cv_results['test_precision'].mean(),
                     cv_results['test_recall'].mean()
                     )
    #for score_name in score_list:
        #print(f"{score_name}: {round(cv_results['test_'+ score_name].mean(), 4)} ({model_name}) ")
model_metrics
## stratifier impact
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify = y)

print("Base Models....")
classifiers = [
               ("RF_withstratify", RandomForestClassifier()),
               ('XGBoost_withstratify', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
               ]

score_list = ["roc_auc", "f1", "precision", "recall", "accuracy"]
for model_name, classifier in classifiers:
    cv_results = cross_validate(classifier , X_train, y_train, cv=3, scoring=score_list)
    model_metrics = add_model_metric(model_name + ' stratify',
                                     cv_results['test_roc_auc'].mean(),
                                     cv_results['test_accuracy'].mean(),
                                     cv_results['test_f1'].mean(),
                                     cv_results['test_precision'].mean(),
                                     cv_results['test_recall'].mean()
                                     )

    model = classifier.fit(X_train,y_train)
    model_metrics = add_model_metric(model_name + "test",
                     roc_auc_score(y_test, model.predict(X_test)),
                     accuracy_score(y_test,model.predict(X_test)),
                     f1_score(y_test, model.predict(X_test)),
                     0,0
                     )

model_metrics



print("Hyperparameter Optimization....")
    classifiers = [
                   ("RF", RandomForestClassifier(),rf_params),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'),xgboost_params),
                   ]
    cv = 3
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        score_list = ["roc_auc", "f1", "precision", "recall", "accuracy"]
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=score_list)
        model_metrics = add_model_metric(name + ' After Hyperparameter Optimization' ,
                         cv_results['test_roc_auc'].mean(),
                         cv_results['test_accuracy'].mean(),
                         cv_results['test_f1'].mean(),
                         cv_results['test_precision'].mean(),
                         cv_results['test_recall'].mean()
                         )
        best_models[name] = final_model
        model_metrics.sort_values("Model")
        print("Voting Classifier...")
        voting_clf = VotingClassifier(estimators=[('XGBoost', best_models["XGBoost"]),
                                                  ('RF', best_models["RF"])],
                                  voting='soft').fit(X, y)
        cv_results = cross_validate(voting_clf, X, y, cv=cv, scoring=score_list)

        model_metrics = add_model_metric('Voting Classifier',
                         cv_results['test_roc_auc'].mean(),
                         cv_results['test_accuracy'].mean(),
                         cv_results['test_f1'].mean(),
                         cv_results['test_precision'].mean(),
                         cv_results['test_recall'].mean()
                         )
        model_metrics.sort_values("F1",ascending=False)
## plot İmportance
        model = XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=0.5, early_stopping_rounds=None,
              enable_categorical=False, eval_metric='logloss',
              feature_types=None, gamma=None, gpu_id=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.01, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=8,
              max_leaves=None, min_child_weight=None, missing=None,
              monotone_constraints=None, n_estimators=100, n_jobs=None,
              num_parallel_tree=None, predictor=None, random_state=None).fit(X,y)

        features = X
        num = len(X)
        save=True
        feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
        plt.title('Features')
        plt.tight_layout()
        plt.show(block=True)
        if save:
            plt.savefig('importances.png')

# Explain model predictions using shap library:
X_importance= X
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_importance)
shap.summary_plot(shap_values, X_importance, plot_type=None)

X_interaction = X_importance.iloc[:500,:]
shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_interaction)
shap.summary_plot(shap_interaction_values, X_interaction)
smple = X[X.index == 1]
shap.plots.waterfall(shap_values[1], max_display=14)

explainer = shap.Explainer(model, X)
shap_values = explainer(smple)
shap.summary_plot(shap_values, smple, feature_names=X.columns)