import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
import tensorflow as tf
import joblib

app = Flask(__name__)

sc = joblib.load('models/scaler.pkl')

# model = pickle.load(open('models/scaler.pkl', 'rb'))
model = tf.keras.models.load_model('models/model.h5')
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # const=float(request.form.get('const',False))
        age = float(request.form.get('age', False))
        sex = float(request.form.get('sex', False))
        onthyroxine = float(request.form.get('onthyroxine', False))
        queryonthyroxine = float(request.form.get('queryonthyroxine', False))
        onantithyroidmedication = float(request.form.get('onantithyroidmedication', False))
        sick = float(request.form.get('sick', False))
        pregnant = float(request.form.get('pregnant', False))
        thyroidsurgery = float(request.form.get('thyroidsurgery', False))
        I131treatment = float(request.form.get('I131treatment', False))
        queryhypothyroid = float(request.form.get('queryhypothyroid', False))
        queryhyperthyroid = float(request.form.get('queryhyperthyroid', False))
        lithium = float(request.form.get('lithium', False))
        goitre = float(request.form.get('goitre', False))
        tumor = float(request.form.get('tumor', False))
        hypopituitary = float(request.form.get('hypopituitary', False))
        psych = float(request.form.get('psych', False))
        TSH_measured = float(request.form.get('TSH measured' , False))
        TSH = float(request.form.get('TSH', False))
        T3_measured = float(request.form.get('T3 measured', False))
        T3 = float(request.form.get('T3', False))
        TT4_measured = float(request.form.get('TT4 measured' , False))
        TT4 = float(request.form.get('TT4' , False))
        T4U_measured = float(request.form.get('T4U measured', False))
        T4U = float(request.form.get('T4U', False))
        FTI_measured = float(request.form.get('FTI measured', False))
        FTI = float(request.form.get('FTI', False))
        TBG_measured = float(request.form.get('FTI measured', False))

    # values = ({"age": age, "sex": sex,
    #          "TSH": TSH, "T3": T3, "T4U": T4U, "FTI": FTI,
    #         "onthyroxine": onthyroxine, "queryonthyroxine": queryonthyroxine,
    #         "onantithyroidmedication": onantithyroidmedication,
    #         "sick": sick, "pregnant": pregnant, "thyroidsurgery": thyroidsurgery,
    #         "I131treatment": I131treatment,
    #         "queryhypothyroid": queryhypothyroid, "queryhyperthyroid": queryhyperthyroid,
    #         "lithium": lithium, "goitre": goitre, "tumor": tumor,
    #         "hypopituitary": hypopituitary,
    #         "psych": psych})
    
        # "const" : const,
    values = ({
        "age" : age , "sex" :sex , 'on thyroxine':onthyroxine,'query on thyroxine':queryonthyroxine,
        "on antithyroid medication" : onantithyroidmedication , 'sick':sick , 'pregnant' :pregnant,
        'thyroid surgery':thyroidsurgery , 'I131 treatment':I131treatment , 'query hypothyroid':queryhypothyroid,
        'query hyperthyroid':queryhyperthyroid , 'lithium':lithium , 'goitre':goitre, 'tumor':tumor, 'hypopituitary':hypopituitary, 'psych':psych, 'TSH':TSH,
        'T4U':T4U,'FTI':FTI,'TT4':TT4,'T3':T3,'T3 measured':T3_measured,  'TT4 measured':TT4_measured, 'T4U measured':T4U_measured, 
        'FTI measured':FTI_measured, 'TBG measured':TBG_measured ,'TSH measured':TSH_measured
        })
        # values = ({"age": age, "sex": sex,
        #        "TSH": TSH, "T3": T3, "T4U": T4U, "FTI": FTI,
        #        "onthyroxine": onthyroxine, "queryonthyroxine": queryonthyroxine,
        #        "onantithyroidmedication": onantithyroidmedication,
        #        "sick": sick, "pregnant": pregnant, "thyroidsurgery": thyroidsurgery,
        #        "I131treatment": I131treatment,
        #        "queryhypothyroid": queryhypothyroid, "queryhyperthyroid": queryhyperthyroid,
        #        "lithium": lithium, "goitre": goitre, "tumor": tumor,
        #        "hypopituitary": hypopituitary,
        #        "psych": psych})
    df_transform = pd.DataFrame.from_dict([values])
    # print("applying transformation\n")

    df_transform.age = df_transform['age'] ** (1 / 2)
    # print(df_transform.age)

    df_transform.TSH = np.log1p(df_transform['TSH'])

    df_transform.T3 = df_transform['T3'] ** (1 / 2)

    df_transform.T4U = np.log1p(df_transform['T4U'])

    df_transform.FTI = df_transform['FTI'] ** (1 / 2)

    

    df_transform.to_dict()

        # const,
    arr =np.array([[
                     df_transform.age, sex,
                    onthyroxine, queryonthyroxine,
                    onantithyroidmedication,
                    sick, pregnant, thyroidsurgery,
                    I131treatment,
                    queryhypothyroid, queryhyperthyroid,
                    lithium, goitre, tumor,
                    hypopituitary,df_transform.TSH, df_transform.T3,df_transform.T4U  , df_transform.FTI,
                    psych , TSH_measured,  T3_measured ,TT4_measured , df_transform.TT4 , T4U_measured  , FTI_measured ,
                    TBG_measured]],dtype=object)
    # print("After transformation:\n")
    # print(arr)
    pred = np.argmax(model.predict(sc.transform(arr)))

    if pred == 0:
        res_Val = "Negative"
    else :
        res_Val = "positive"
        

    return render_template('result.html', predictions=(res_Val))

if __name__ == "__main__":
    app.run(debug=True)