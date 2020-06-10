from IPython import get_ipython
import pandas as pd
from pandas_profiling import ProfileReport

from sklearn.feature_selection import SelectKBest,RFE,f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,Lasso,ElasticNet,LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,LabelEncoder,OneHotEncoder,Normalizer
from sklearn.metrics import mean_squared_error
from math import sqrt

get_ipython().run_line_magic('matplotlib','qt')

names=['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style',
       'drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size',
       'fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
df = pd.read_csv('imports-85-data.csv',names=names)

#df.plot(kind='density',subplots=True,layout=(4,3),sharex=False,sharey=False,fontsize=15)

#profile = ProfileReport(df)
#profile.to_file('auto.html')

#df.drop('symboling',axis=1,inplace=True)
#df.drop('make',axis=1,inplace=True)
#df.drop('engine-location',axis=1,inplace=True)
#df.drop('normalized-losses',axis=1,inplace=True)
df.drop(df.loc[df['num-of-doors']=='?'].index, inplace=True)
df.drop(df.loc[df['bore']=='?'].index, inplace=True)
df.drop(df.loc[df['stroke']=='?'].index, inplace=True)
df.drop(df.loc[df['normalized-losses']=='?'].index, inplace=True)

new_df = df[['normalized-losses','make','num-of-doors','body-style','wheel-base','length','width','height','curb-weight','engine-type','engine-size','compression-ratio','peak-rpm','city-mpg','highway-mpg']]

array = new_df.values
x = array[:,:-1]
y = array[:,-1]

oe = OrdinalEncoder()

new_x = oe.fit_transform(x)

le = LabelEncoder()

new_y = le.fit_transform(y)

model = LogisticRegression(solver='liblinear')
rfe = RFE(model, 15)
fit = rfe.fit(new_x, new_y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
'''
pca = PCA(n_components=15)
fit = pca.fit(new_x)
print(fit.components_)


model = ExtraTreesClassifier(n_estimators=100)
model.fit(new_x,new_y)
print(model.feature_importances_)
'''

x_train,x_test,y_train,y_test = train_test_split(new_x,new_y,test_size=0.20,random_state=7)


models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
print('\t')

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',
Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',
ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
print('\t')
    
pipelines = []
pipelines.append(('NormalizedLR', Pipeline([('Normalizer', Normalizer()),('LR',
LinearRegression())])))
pipelines.append(('NormalizedLASSO', Pipeline([('Normalizer', Normalizer()),('LASSO',
Lasso())])))
pipelines.append(('NormalizedEN', Pipeline([('Normalizer', Normalizer()),('EN',
ElasticNet())])))
pipelines.append(('NormalizedKNN', Pipeline([('Normalizer', Normalizer()),('KNN',
KNeighborsRegressor())])))
pipelines.append(('NormalizedCART', Pipeline([('Normalizer',Normalizer()),('CART',
DecisionTreeRegressor())])))
pipelines.append(('NormalizedSVR', Pipeline([('Normalizer', Normalizer()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)






print('\t')
model = LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(sqrt(mean_squared_error(y_test,predictions)))
print('\t')

model = ElasticNet()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(sqrt(mean_squared_error(y_test,predictions)))
print('\t')

model = DecisionTreeRegressor()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(sqrt(mean_squared_error(y_test,predictions)))
