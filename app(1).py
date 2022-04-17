#!/usr/bin/env python
# coding: utf-8

# ### Libraries

# In[28]:


from flask import Flask, request, render_template
import joblib
import pandas as pd


# ### App

# In[29]:


app = Flask(__name__)


# In[30]:


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = request.form.get("age")
        limit_bal = request.form.get("limitbal")
        pay_0 = request.form.get("pay0")
        bill_amt1 = request.form.get("billamt1")
  
        print(age, limit_bal, pay_0, bill_amt1)
    
        age = float(age)
        limit_bal = float(limit_bal)
        pay_0 = float(pay_0)
        bill_amt1 = float(bill_amt1)
              
        model1 = joblib.load("logistic_regression")
        pred1 = model1.predict([[age, limit_bal, pay_0, bill_amt1]])
        model2 = joblib.load("decision_tree")
        pred2 = model2.predict([[age, limit_bal, pay_0, bill_amt1]])
        model3 = joblib.load("random_forest")
        pred3 = model3.predict([[age, limit_bal, pay_0, bill_amt1]])
        model4 = joblib.load("xgboost")
        pred4 = model4.predict(pd.DataFrame([[age, limit_bal, pay_0, bill_amt1]]))
        
        return(render_template("index1.html", result1 = pred1, result2 = pred2, result3 = pred3, result4 = pred4 )) 
    else:
        return(render_template("index1.html", result1 = "Enter values for prediction", result2 = "Enter values for prediction", result3 = "Enter values for prediction", result4 = "Enter values for prediction")) 


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




