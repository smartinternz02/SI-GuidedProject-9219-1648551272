#import the libraries
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
model=load_model("text_generation_rnn.hdf5",compile=False)

app=Flask(__name__,template_folder="templates")
@app.route('/')
def welcome():
    return render_template('home.html')
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['GET','POST'])
def pred():
    if request.method=='POST':
        initial_text=request.form['message']
        initial_text=initial_text.lower()
        chars=['\n', ' ', '!', '$', '%', '(', ')', ',', '-', '.',
               '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', 
               '9', ':', ';', '?', '[', ']', 'a', 'b', 'c', 'd', 
               'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
               'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
               'y', 'z', '‘', '’', '“', '”']
        char_to_int = dict((c, i) for i, c in enumerate(chars))
        initial_text = [char_to_int[c] for c in initial_text]
        int_to_char = dict((i, c) for i, c in enumerate(chars))
        test_text = initial_text
        generated_text = []
        SEQ_LENGTH = 100
        VOCABULARY = len(chars)
        if len(test_text) > 100:
            test_text = test_text[len(test_text)-100: ]
        if len(test_text) < 100:
            pad = []
            space = char_to_int[' ']
            pad = [space for i in range(100-len(test_text))]
            test_text = pad + test_text
        for i in range(100):
            X = np.reshape(test_text, (1, SEQ_LENGTH, 1))
            X  = X  / float(VOCABULARY)
            Prediction = model.predict(X)
            index = np.argmax(Prediction)
            result = int_to_char[index]
            generated_text.append(result)
            test_text.append(index)
            test_text = test_text[1:]
        output=''.join(generated_text)
        return render_template('predict.html',prediction=(output))
    else:
        return render_template('predict.html')
if __name__ == '__main__':
    app.run(host='localhost',port=5000,debug=False,threaded=False)