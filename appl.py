import flask
import joblib
import numpy as np
from flask import render_template
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor


appl=flask.Flask(__name__,template_folder='template')

@appl.route('/', methods=['POST','GET'])

@appl.route('/index', methods=['POST','GET'])

def main():
    if flask.request.method=='GET':
        return render_template('main.html')

    if flask.request.method=='POST':
        with open('GradientBoostingRegressor.pkl','rb') as f:
            model=joblib.load(f)
        dats=[float(flask.request.form['IW']),
            float(flask.request.form['FI']),
            float(flask.request.form['VW']),
            float(flask.request.form['FP'])]
        predict=np.round(model.predict([dats]),2)
        texst="Входные параметры IW:"+str(round(dats[0],2))+",  FI:"+str(round(dats[1],2))+", VW:"+str(round(dats[2],2))+", FP: "+str(round(dats[3],2))
        res=predict.flatten().tolist()
        return render_template('main.html',result=res, main_text=texst)

if __name__=='__main__':
    appl.run(debug=False)
