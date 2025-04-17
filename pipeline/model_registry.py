from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def get_classifiers():
    return {
        'XGBoost': XGBClassifier(eval_metric='logloss'),
        'LASSO': LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000),
        #'ElasticNet': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000),
        #'KNN': KNeighborsClassifier(),
        #'SVM': SVC(probability=True),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'ExtraTrees': ExtraTreesClassifier(),
        #'MLP': MLPClassifier(max_iter=1000)
    }
