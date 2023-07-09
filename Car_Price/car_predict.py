import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from flask import Flask, request, jsonify
import xgboost as xgb

pd.set_option('display.max_columns', None)
data = pd.read_csv("autos.csv")
data.drop(columns=["index", "dateCrawled","abtest","name","seller","offerType","monthOfRegistration","dateCreated","nrOfPictures","postalCode","lastSeen"], inplace=True)

def cleaning(data):
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data = data[data.price != 0]
    data = data[data.powerPS != 0]
    data.reset_index(inplace=True,drop=True)
    return data

data = cleaning(data)

def aykiri_deger_silme(data,i):
    Q1 = data[i].quantile(0.25)
    Q3 = data[i].quantile(0.75)
    IQR = Q3-Q1
    alt_sinir= Q1 - 1.5*IQR
    üst_sinir= Q1 + 1.5*IQR
    if i == 'price':
        alt_sinir = 99
    elif i == 'powerPS':
        alt_sinir = 50
    aykiri_deger=(data[i] < alt_sinir) | (data[i] > üst_sinir)
    data[i]=data[i][~aykiri_deger]
    data.dropna(inplace=True)
    data.reset_index(inplace=True,drop=True)
    return data

data=aykiri_deger_silme(data,'powerPS')
data=aykiri_deger_silme(data,'yearOfRegistration')
data=aykiri_deger_silme(data,'price')

#Label Encoder
le=LabelEncoder()
data['gearbox'] = le.fit_transform(data['gearbox'])
data['notRepairedDamage'] = le.fit_transform(data['notRepairedDamage'])

#Çok değişkenlilere one hot
ohe = OneHotEncoder()
def get_ohe(df,i):
    new_arrays=ohe.fit_transform(df[[i]]).toarray()
    our_labels=np.array(ohe.categories_).ravel()
    ohe_df=pd.DataFrame(new_arrays, columns=our_labels)
    df.drop(columns=[i], axis=1, inplace=True)
    new_df=pd.concat([df,ohe_df],axis=1)
    return new_df

data=get_ohe(data,'vehicleType')
data=get_ohe(data,'fuelType')
data=get_ohe(data,'brand')
data=get_ohe(data,'model')

# Model
X=data.drop('price',axis=1).values
y=data['price'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=33)

model = xgb.XGBRegressor(max_depth=5,tree_method="hist", n_estimators = 650).fit(X_train, y_train)
pred = model.predict(X_test)
#print(f"R2 Score : {r2_score(y_test, pred)}")

def tahmin(test):
    pd.set_option('display.max_columns', None)
    data = pd.read_csv("autos.csv")
    data.drop(columns=["index", "dateCrawled","abtest","name","seller","offerType","monthOfRegistration","dateCreated","nrOfPictures","postalCode","lastSeen"], inplace=True)

    def cleaning(data):
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        data = data[data.price != 0]
        data = data[data.powerPS != 0]
        data.reset_index(inplace=True,drop=True)
        return data

    data = cleaning(data)
    
    def get_ohe(df,i,test):
        new_arrays=ohe.fit_transform(df[[i]]).toarray()
        our_labels=np.array(ohe.categories_).ravel()
        ohe_df=pd.DataFrame(new_arrays, columns=our_labels)
        df.drop(columns=[i], axis=1, inplace=True)
        new_df=pd.concat([df,ohe_df],axis=1)
        
        new_arrays=ohe.transform(test[[i]]).toarray()
        our_labels=np.array(ohe.categories_).ravel()
        ohe_df=pd.DataFrame(new_arrays, columns=our_labels)
        test.drop(columns=[i], axis=1, inplace=True)
        test=pd.concat([test,ohe_df],axis=1)
        return new_df,test
 
    data,test=get_ohe(data,'vehicleType',test) 
    data,test=get_ohe(data,'fuelType',test)
    data,test=get_ohe(data,'brand',test)
    data,test=get_ohe(data,'model',test)
    
    data['gearbox'] = le.fit_transform(data['gearbox'])
    test['gearbox'] = le.transform(test['gearbox'])
    data['notRepairedDamage'] = le.fit_transform(data['notRepairedDamage'])
    test['notRepairedDamage'] = le.transform(test['notRepairedDamage'])
    
    test.drop(['trabant', '601','911','b_max','boxster','cayenne','discovery_sport','gl','q7','range_rover_evoque','rangerover','serie_2','serie_3'], axis=1,inplace=True)
    a = model.predict(test)
    
    return a

def test_degerleri(a,b,c,d,e,f,g,h,i):
    x = {'vehicleType':[a],'yearOfRegistration':[b],'gearbox':[c],'powerPS':[d],'model':[e],'kilometer':[f],'fuelType':[g],'brand':[h],'notRepairedDamage':[i]}
    test=pd.DataFrame.from_dict(x)
    res=tahmin(test)
    return str(res[0])

app = Flask(__name__) 
    
@app.route('/car_predict', methods = ['POST'])

def upload_page():
    try:
        a = request.json['vehicleType']
        b = request.json['yearOfRegistration']
        c = request.json['gearbox'] 
        d = request.json['powerPS']
        e = request.json['model']
        f = request.json['kilometer']
        g = request.json['fuelType']
        h = request.json['brand']
        i = request.json['notRepairedDamage']
        return jsonify(test_degerleri(a,b,c,d,e,f,g,h,i))
    
    except Exception as y:
        response={}
        response['ErrorMessage'] = str(y)
        response['HasFailed']= True
        return response
    
if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)

