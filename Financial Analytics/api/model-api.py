import numpy as np
import requests
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)

CORS(app)

model = pickle.load(open('../models/model.pkl', 'rb'))
family_min= 1.0
income_min = 27000.0
demployed_min =  17.0
age_min = 21.0

family_max= 20.0
income_max = 1575000.0
demployed_max=  15713.0
age_max = 69.0

features_col = ['NAME_INCOME_TYPE==Commercial associate', 'NAME_INCOME_TYPE==Pensioner',
       'NAME_INCOME_TYPE==State servant', 'NAME_INCOME_TYPE==Student',
       'NAME_INCOME_TYPE==Working', 'NAME_EDUCATION_TYPE==Academic degree',
       'NAME_EDUCATION_TYPE==Higher education',
       'NAME_EDUCATION_TYPE==Incomplete higher',
       'NAME_EDUCATION_TYPE==Lower secondary',
       'NAME_EDUCATION_TYPE==Secondary / secondary special',
       'NAME_FAMILY_STATUS==Civil marriage', 'NAME_FAMILY_STATUS==Married',
       'NAME_FAMILY_STATUS==Separated',
       'NAME_FAMILY_STATUS==Single / not married', 'NAME_FAMILY_STATUS==Widow',
       'NAME_HOUSING_TYPE==Co-op apartment',
       'NAME_HOUSING_TYPE==House / apartment',
       'NAME_HOUSING_TYPE==Municipal apartment',
       'NAME_HOUSING_TYPE==Office apartment',
       'NAME_HOUSING_TYPE==Rented apartment',
       'NAME_HOUSING_TYPE==With parents', 'OCCUPATION_TYPE==Accountants',
       'OCCUPATION_TYPE==Cleaning staff', 'OCCUPATION_TYPE==Cooking staff',
       'OCCUPATION_TYPE==Core staff', 'OCCUPATION_TYPE==Drivers',
       'OCCUPATION_TYPE==HR staff', 'OCCUPATION_TYPE==High skill tech staff',
       'OCCUPATION_TYPE==IT staff', 'OCCUPATION_TYPE==Laborers',
       'OCCUPATION_TYPE==Low-skill Laborers', 'OCCUPATION_TYPE==Managers',
       'OCCUPATION_TYPE==Medicine staff', 'OCCUPATION_TYPE==Pensioner',
       'OCCUPATION_TYPE==Private', 'OCCUPATION_TYPE==Private service staff',
       'OCCUPATION_TYPE==Realty agents', 'OCCUPATION_TYPE==Sales staff',
       'OCCUPATION_TYPE==Secretaries', 'OCCUPATION_TYPE==Security staff',
       'OCCUPATION_TYPE==Waiters/barmen staff', 'AMT_INCOME_TOTAL',
       'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'AGE', 'FLAG_OWN_REALTY1']


def predict_func(dic):
    features = [float(dic[feature]) for feature in features_col]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    # print(str(prediction[0]))
    # print(type(str(prediction[0])))
    if str(prediction[0]) == '0':
        result = "Good"
        # print("here")
    elif str(prediction[0]) == '1':
        result = "Bad"
        # print("bad here")
    else:
        result = "sus"
        print(type(prediction))
    # print(dic['date'])
    return {'prediction': result}

# # run the model to predict current, hourly forecast, daily forecast values

@app.route("/predict" , methods=['POST'])
@cross_origin()
def process_form() :
    data = request.get_json()
    print(data)
    dic = {}
    dic['AGE'] = 0
    dic['NAME_FAMILY_STATUS==Civil marriage'] = 0
    dic['NAME_FAMILY_STATUS==Married'] = 0 
    dic['NAME_FAMILY_STATUS==Separated'] = 0
    dic['NAME_FAMILY_STATUS==Single / not married'] = 0
    dic['NAME_FAMILY_STATUS==Widow'] = 0

    dic['OCCUPATION_TYPE==Accountants'] = 0
    dic['OCCUPATION_TYPE==Cleaning staff'] = 0
    dic['OCCUPATION_TYPE==Cooking staff'] = 0
    dic['OCCUPATION_TYPE==Core staff'] = 0
    dic['OCCUPATION_TYPE==Drivers'] = 0
    dic['OCCUPATION_TYPE==HR staff'] = 0
    dic['OCCUPATION_TYPE==High skill tech staff'] = 0
    dic['OCCUPATION_TYPE==IT staff'] = 0
    dic['OCCUPATION_TYPE==Laborers'] = 0
    dic['OCCUPATION_TYPE==Low-skill Laborers'] = 0 
    dic['OCCUPATION_TYPE==Managers'] = 0
    dic['OCCUPATION_TYPE==Medicine staff'] = 0
    dic['OCCUPATION_TYPE==Pensioner'] = 0
    dic['OCCUPATION_TYPE==Private'] = 0
    dic['OCCUPATION_TYPE==Private service staff'] = 0
    dic['OCCUPATION_TYPE==Realty agents'] = 0
    dic['OCCUPATION_TYPE==Sales staff'] = 0
    dic['OCCUPATION_TYPE==Secretaries'] = 0
    dic['OCCUPATION_TYPE==Security staff'] = 0
    dic['OCCUPATION_TYPE==Waiters/barmen staff'] = 0

    dic['NAME_INCOME_TYPE==Commercial associate'] = 0 
    dic['NAME_INCOME_TYPE==Pensioner'] = 0
    dic['NAME_INCOME_TYPE==State servant'] = 0 
    dic['NAME_INCOME_TYPE==Student'] = 0
    dic['NAME_INCOME_TYPE==Working'] = 0 

    dic['NAME_EDUCATION_TYPE==Academic degree'] = 0
    dic['NAME_EDUCATION_TYPE==Higher education'] = 0
    dic['NAME_EDUCATION_TYPE==Incomplete higher'] = 0
    dic['NAME_EDUCATION_TYPE==Lower secondary'] = 0
    dic['NAME_EDUCATION_TYPE==Secondary / secondary special'] = 0

    dic['NAME_HOUSING_TYPE==Co-op apartment'] = 0
    dic['NAME_HOUSING_TYPE==House / apartment'] = 1
    dic['NAME_HOUSING_TYPE==Municipal apartment'] = 0
    dic['NAME_HOUSING_TYPE==Office apartment'] = 0
    dic['NAME_HOUSING_TYPE==Rented apartment'] = 0
    dic['NAME_HOUSING_TYPE==With parents'] = 0


    dic['DAYS_EMPLOYED'] = 0

    dic['FLAG_OWN_REALTY1'] = 0

    dic['AMT_INCOME_TOTAL'] = 0

    dic['CNT_FAM_MEMBERS'] = 0 


    for key in data.keys():
        if key == 'age':
            X = int(data[key])
            X_std = (X - age_min) / (age_max - age_min)
            X_scaled = X_std * (age_max - age_min) + age_min
            dic['AGE'] = X_scaled
        elif key == 'marriage':
            if data[key] == 'Civil Marriage':
                dic['NAME_FAMILY_STATUS==Civil marriage'] = 1
            elif data[key] == 'Married':
                dic['NAME_FAMILY_STATUS==Married'] = 1
            elif data[key] == 'Seperated':
                dic['NAME_FAMILY_STATUS==Separated'] = 1
            elif data[key] == 'Single / not married':
                dic['NAME_FAMILY_STATUS==Single / not married'] = 1
            else:
                dic['NAME_FAMILY_STATUS==Widow'] = 1
        elif key == 'housing':
            if data[key] == "Academic degree":
                dic['NAME_HOUSING_TYPE==Co-op apartment'] = 1
            elif data[key] == "Higher education":
                dic['NAME_HOUSING_TYPE==House / apartment'] = 1
            elif data[key] == "Incomplete higher":
                dic['NAME_HOUSING_TYPE==Municipal apartment'] = 1
            elif data[key] == "Lower secondary":
                dic['NAME_HOUSING_TYPE==Rented apartment'] = 1
            else:
                dic['NAME_HOUSING_TYPE==With parents'] = 1
        elif key == 'income_type':
            if data[key] == "Commercial associate":
                dic['NAME_INCOME_TYPE==Commercial associate'] = 1
            elif data[key] == "Pensioner":
                dic['NAME_INCOME_TYPE==Pensioner'] = 1
            elif data[key] == "State servant":
                dic['NAME_INCOME_TYPE==State servant'] = 1
            elif data[key] == "Student":
                dic['NAME_INCOME_TYPE==Student'] = 1
            else:
                dic['NAME_INCOME_TYPE==Working']  = 1
        elif key == 'occupation':
            if data[key] == 'Accountants':
                dic['OCCUPATION_TYPE==Accountants'] = 1
            elif data[key] == "Cleaning staff":
                dic['OCCUPATION_TYPE==Cleaning staff'] = 1
            elif data[key] == "Cooking staff":
                dic['OCCUPATION_TYPE==Cooking staff'] = 1
            elif data[key] == "Core staff":
                dic['OCCUPATION_TYPE==Core staff'] = 1
            elif data[key] == "Drivers":
                dic['OCCUPATION_TYPE==Drivers'] = 1
            elif data[key] == "HR staff":
                dic['OCCUPATION_TYPE==HR staff'] = 1
            elif data[key] == "High skill tech staff":
                dic['OCCUPATION_TYPE==High skill tech staff'] = 1
            elif data[key] == "IT staff":
                dic['OCCUPATION_TYPE==IT staff'] = 1
            elif data[key] == "Laborers":
                dic['OCCUPATION_TYPE==Laborers'] = 1
            elif data[key] == "Managers":
                dic['OCCUPATION_TYPE==Managers'] = 1
            elif data[key] == "Medicine staff":
                dic['OCCUPATION_TYPE==Medicine staff'] = 1
            elif data[key] == "Pensioner":
                dic['OCCUPATION_TYPE==Pensioner'] = 1
            elif data[key] == "Private service staff":
                dic['OCCUPATION_TYPE==Private service staff'] = 1
            elif data[key] == "Realty agents":
                dic['OCCUPATION_TYPE==Realty agents'] = 1
            elif data[key] == "Sales staff":
                dic['OCCUPATION_TYPE==Sales staff'] = 1
            elif data[key] == "Secretaries":
                dic['OCCUPATION_TYPE==Secretaries'] = 1
            elif data[key] == "Security staff":
                dic['OCCUPATION_TYPE==Security staff'] = 1
            elif data[key] == "Waiters/barmen staff":
                dic['OCCUPATION_TYPE==Waiters/barmen staff'] = 1
            else:
                dic['OCCUPATION_TYPE==Private'] = 1
        elif key == 'education':
            if data[key] == "Academic degree":
                dic['NAME_EDUCATION_TYPE==Academic degree'] = 1
            elif data[key] == "Higher education":
                dic['NAME_EDUCATION_TYPE==Higher education'] = 1
            elif data[key] == "Incomplete higher":
                dic['NAME_EDUCATION_TYPE==Incomplete higher'] = 1
            elif data[key] == "Lower secondary":
                dic['NAME_EDUCATION_TYPE==Lower secondary'] = 1
            else:
                dic['NAME_EDUCATION_TYPE==Secondary / secondary special'] = 1
        elif key == 'demployed':
            X = int(data[key])
            X_std = (X - demployed_min) / (demployed_max - demployed_min)
            X_scaled = X_std * (demployed_max - demployed_min) + demployed_min
            dic['DAYS_EMPLOYED'] = X_scaled
        elif key == 'realty':
            if data[key] == "Y":
                dic['FLAG_OWN_REALTY1'] = 1
        elif key == 'income':
            X = int(data[key])
            X_std = (X - income_min) / (income_max - income_min)
            X_scaled = X_std * (income_max - income_min) + income_min
            dic['AMT_INCOME_TOTAL'] = X_scaled
        else:
            X = int(data[key])
            X_std = (X - family_min) / (family_max - family_min)
            X_scaled = X_std * (family_max - family_min) + family_min
            dic['CNT_FAM_MEMBERS'] = X_scaled
    

    return jsonify(
        {
            "code": 200,
            "data": {
                "result": predict_func(dic)
            }
        }
    )
    # material = Materials(**data)
    # // for loop
    # put predict inside then return result
    # print(material)
    # try:
    #     db.session.add(material)
    #     db.session.commit()
    #     return jsonify(
    #         {
    #             "code": 200,
    #             "message": "Course has been added successfully.",
    #             "data": [material.to_dict()]
    #         }
    #         ), 200
    # except SQLAlchemyError as e:
    #     print(str(e))
    #     db.session.rollback()
    #     return jsonify({
    #         "code": 500,
    #         "message": "Unable to add material to database."
    #     }), 500
    #scale
    # return distcionary of the column key and the value is value




# get the form
# post the form

if __name__ == "__main__":
    app.run(port=5001, debug=True)