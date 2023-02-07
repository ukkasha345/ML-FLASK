from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('./Model/RandomForest_final.pkl','rb'))


@app.route("/")
def home():
    return render_template('index.html')

@app.route('/pred_diet', methods=['POST'])
def pred_diet():
    vla = [float(x) for x in request.form.values()]
    print(vla)
    scndarr = [np.array(vla)]
    result = model.predict(scndarr)
    # pred = str(model.predict([[1.0,71.730978,220.042470,56,0.0]]))
    health_results=""
    if(result[0]==0.0):
        health_results="You are [HEALTHY]"
    elif(result[0]==1.0):
        health_results="You have to start [Weight Loss ]"    
    else:
        health_results="You have to [Weight Gain]"
    
    return render_template('index.html', output=health_results)

if __name__ == '__main__':
    app.run(debug=True)