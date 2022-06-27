from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

def auto_model(args, params, num_classes, model_name, SEED, key):

    X_train, X_test, y_train, y_test = args

    print(f'training model: {model_name}, key: {key}')    

    gs = GridSearchCV(eval(params[0])(random_state=SEED) if params[2] else eval(params[0])(), params[1],
                      scoring='roc_auc', cv=3, n_jobs=-1, verbose=4)
    gs.fit(X_train, y_train)
    bestParams = gs.best_params_
    print(f'{model_name} 模型最优参数: {bestParams}\n')
    
    y_pred = gs.predict(X_test)
    try:
        y_score = gs.predict_proba(X_test)
    except:
        y_score = gs.best_estimator_._predict_proba_lr(X_test)

    if num_classes == 2:
        y_score = y_score[:, 1]
        AUC = metrics.roc_auc_score(y_test, y_score)
    else:
        AUC = metrics.roc_auc_score(y_test, y_score, multi_class='ovr')

    return dict(
        AUC=AUC,
        y_test=y_test.astype('int').tolist(),
        y_pred=y_pred.astype('int').tolist(),
        y_score=y_score.tolist(),
        bestParams=bestParams
    )
