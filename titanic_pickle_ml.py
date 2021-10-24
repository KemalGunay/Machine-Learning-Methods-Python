
# libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
import pickle


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



def load():
    data = pd.read_pickle(r"C:\Users\Kemal\Desktop\veri_bilimi\VBO_BOOTCAMP\HAFTA_06\HAFTA_06\Ders Notları\titanic_data_prep.pkl")
    return data

df=load()


df.head()

# Model & Prediction
######################################################

# Bağımlı ve bağımsız değişkelerin seçilmesi:
y = df["SURVIVED"]
X = df.drop(["SURVIVED", "PASSENGERID"], axis=1)


log_model = LogisticRegression().fit(X, y)


#  LOGISTIC REGRESSION PREDICTION
y_pred = log_model.predict(X)



# Başarı skorları:
print(classification_report(y, y_pred))


# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)





# Holdout Yöntemi
# Veri setinin train-test olarak ayrılması:
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)


# Modelin train setine kurulması:
log_model = LogisticRegression().fit(X_train, y_train)

# Test setinin modele sorulması:
y_pred = log_model.predict(X_test)

# AUC Score için y_prob (1. sınıfa ait olma olasılıkları)
y_prob = log_model.predict_proba(X_test)[:, 1]


# Classification report
print(classification_report(y_test, y_pred))


# ROC Curve
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

roc_auc_score(y_test, y_prob)




# Cross Validation
# gözlem sayısı az olduğu için 3 katlı yaptım
cv_results = cross_validate(log_model,
                            X, y,
                            cv=3,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
# Accuracy: 0.83

cv_results['test_precision'].mean()
# Precision: 0.7992

cv_results['test_recall'].mean()
# Recall: 0.7543

cv_results['test_f1'].mean()
# F1-score: 0.7738

cv_results['test_roc_auc'].mean()
# AUC: 0.8598



# Hyperparameter Tuning -- Grid Search -- Cross Validation
# Decision Tree, SVM, RANDOM FOREST, KNN, LOGISTIC REGRESSION
random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]



cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(X,y)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])

# MODEL BAŞARI ORANLARINI BAR PLOT İLE KARŞILAŞTIRMA
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")

