from flask import *
import pandas as pd
from sklearn.linear_model import LogisticRegression

app=Flask(__name__)
####################################
url = "https://raw.githubusercontent.com/sarwansingh/Python/master/ClassExamples/data/iris.csv"
heading = [ 'Sepallength' , 'Sepalwidth', 'Petallength', 'Petalwidth','species' ]
iris = pd.read_csv(url ,header=None , names=heading)

iris.isnull().sum()

X= iris.iloc[:,:4].values
Y= iris.iloc[:,4].values

modeliris = LogisticRegression()
modeliris.fit(X,Y)
res=modeliris.predict([[ 5.1,3.5,1.4,0.2]])
op=str(res)
###################################

@app.route('/')
def hello_world():
  return render_template("index.html")
  # return 'HELLO AIML FOR PYTHON WEB DEVELOPMENT !!' + op  

@app.route('/project')
def project  ():
  return render_template("form.html") 
 
@app.route('/predict' , methods=['POST'])
def predict  ():
  sl=int(request.form['sl'])
  sw=int(request.form['sw'])
  pl=int(request.form['pl'])
  pw=int(request.form['pw'])  
  
  res=modeliris.predict([[ sl,sw,pl,pw]])
  op="Predicted Flower :"+str(res)
  return render_template("form.html",result=op) 

@app.route('/home')
def home():
  return render_template("index.html")

if __name__=='__main__':
  app.run()       