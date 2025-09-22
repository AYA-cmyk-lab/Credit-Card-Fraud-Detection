#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pickle


# In[ ]:


df = pd.read_csv(r"E:\credit card froud detection dataset\archive\creditcard.csv")
st.write(df.head())
st.write(df.tail())


# In[ ]:


st.write(df.info())


# In[ ]:


st.write(df.describe())


# In[ ]:


st.write(df.isnull().sum())


# In[32]:


#split data
x = df.drop("Class", axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


# In[ ]:


#create and train Random Forest Model
rf = RandomForestClassifier(random_state=42, max_depth=5, n_estimators=100, class_weight="balanced" )
rf.fit(x_train, y_train)

#make predictions
y_pred_rf = rf.predict(x_test)

#evaluate the model
st.write("Random Forest results:")
st.write(f"Accuracy:{accuracy_score(y_test,y_pred_rf):.4f}")
st.write("\nclassification report:")
st.write(classification_report(y_test,y_pred_rf))

#plot feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
n_features = len(importances)
#n_features = len(importances)
plt.figure(figsize=(10,5))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(n_features),importances[indices])
plt.xticks(range(n_features), [f'f(i)' for i in indices], rotation=90)
plt.tight_layout()
st.pyplot(plt)


# In[ ]:


#create and train XGBoost model
xg = xgb.XGBClassifier(n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)
xg.fit(x_train, y_train)

#make predictions
y_pred_xg = xg.predict(x_test)

#evaluate the model
st.write("XGBoost Model Results:")
st.write(f"Accuracy,{accuracy_score(y_test, y_pred_xg):.4f}")
st.write("\nclassification report:")
st.write(classification_report(y_test,y_pred_xg))

#plot confusion matrix
cm = confusion_matrix(y_test, y_pred_xg)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost')
st.pyplot(plt)


# In[ ]:


#create and train Isolation Forest model
isf = IsolationForest(contamination=0.001, random_state=42)
isf.fit(x_train)

#make predictions
y_pred_isf = isf.predict(x_test)
y_pred_isf = [1 if val == -1 else 0 for val in y_pred_isf]
#evaluate the model
st.write("Isolation Forest Results:")
st.write(f"Accuracy,{accuracy_score(y_test, y_pred_isf):.4f}")
st.write("\nclassification report:")
st.write(classification_report(y_test,y_pred_isf))


#plot confusion matrix
cm = confusion_matrix(y_test, y_pred_isf)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Isolation Forest')
st.pyplot(plt)


# In[38]:


#scale the features
scaler= StandardScaler()
x_train_scaled= scaler.fit_transform(x_train)
x_test_scaled= scaler.transform(x_test)


# In[ ]:


#create and train logistic regression model
lr = LogisticRegression()
lr.fit(x_train_scaled, y_train)

#make predictions
y_pred_lr = lr.predict(x_test_scaled)

#evaluate the model
st.write("Logistic regression result:")
st.write(f"accuracy: {accuracy_score(y_test,y_pred_lr):.4f}")
st.write("\nclassification report:")
st.write(classification_report(y_test,y_pred_lr))

#plot confusion matrix
cm= confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion matrix - logistic regression')
plt.xlabel('predicted label')
plt.ylabel('True label')
st.pyplot(plt)

