import numpy as np
from flask import Flask,request,render_template
import pickle
from nltk.corpus import stopwords
stopwords=list(set(stopwords.words('english')))
from fuzzywuzzy import fuzz 



app=Flask(__name__)
model=pickle.load(open('dqi.pickle','rb'))

@app.route('/')
def home():
    return render_template('index4.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    f_list=[]
    sent1=request.form['sent1']
    sent2=request.form['sent2']
    
    s1_len=len(sent1.split())
    s2_len=len(sent2.split())
    
    len_def=abs(s1_len-s2_len)
    f_list.append(len_def)
    
    total_words=s1_len+s2_len
    f_list.append(total_words)
    
    words1=sent1.split()
    words2=sent2.split()
    
    common_token=len(set(words1).intersection(words2))
    f_list.append(common_token)
    
    word_share=round(float(common_token/total_words),2) 
    f_list.append(word_share)
    
    l3=set(words1).intersection(words2)
    stop_count=len(l3.intersection(stopwords))
    f_list.append(stop_count)
    common_word=len(l3.difference(stopwords))
    f_list.append(common_word)
    
    cwc_min=round(float(common_word/(min(s1_len,s2_len))),2)
    f_list.append(cwc_min)
    cwc_max=round(float(common_word/(max(s1_len,s2_len))),2) 
    f_list.append(cwc_max)
    
    csc_min=round(float(stop_count/(min(s1_len,s2_len))),2)
    f_list.append(csc_min)
    csc_max=round(float(stop_count/(max(s1_len,s2_len))),2)
    f_list.append(csc_max)
    
    ctc_min=round(float(common_token/(min(s1_len,s2_len))),2)
    f_list.append(ctc_min)
    ctc_max=round(float(common_token/(max(s1_len,s2_len))),2)
    f_list.append(ctc_max)
    
    if words1[-1]==words2[-1]:
        last_word_freq=1
    else:
        last_word_freq=0
    f_list.append(last_word_freq)
        
    if words1[0]==words2[0] :
        first_word_freq=1
    else:
        first_word_freq=0
    f_list.append(first_word_freq)
        
    fuzz_ratio=fuzz.ratio(str(sent1),str(sent2))
    f_list.append(fuzz_ratio)
    
    partial_ratio=fuzz.partial_ratio(str(sent1),str(sent2))
    f_list.append(partial_ratio)
    
    token_sort_ratio=fuzz.token_sort_ratio(str(sent1),str(sent2))
    f_list.append(token_sort_ratio)
    
    final_features=[np.array(f_list)]
    
    prediction=model.predict(final_features)
    output=prediction[0];
    output=1-output
    
    if output==1:
        output="Sentences are similar"
    else:
        output="Sentences are not similar"
    
    
    return render_template('index4.html',prediction_text=output)

if __name__ == '__main__':
    app.run(debug=True)
    
    
 