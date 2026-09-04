import streamlit as st
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score,accuracy_score,mean_squared_error,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Mini Project",
    layout="wide",
    page_icon=""
)

st.title("ML Model Trainer & Evaluator")
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if st.button("Generate sample data:"):
            data = {
                'Student':['A','B','C','D','E','F'],
                'StudyHours':[6,5,2,3,5,3],
                'sleep':[5,6,6,5,4,7],
                'Marks':[85,90,34,45,89,33]
            }
            df = pd.DataFrame(data)
            st.table(df)

            csv = df.to_csv(index=False)
            st.download_button(
                'Download CSV',
                csv,
                'marks.csv'
            )

if uploaded_file:
    df=pd.read_csv(uploaded_file)
    df=df.dropna()
    df=pd.get_dummies(df)
    st.write(df.head())
    
    task = st.selectbox("Select task", ["Regression", "Classification"])
    
    if task=="Regression":
        RG_model=st.radio("Choose model",["LinearRegression","Polynomial"])
        
        if RG_model=="Polynomial":
            d = st.radio("Degree",[1,2,3], key="degree")
            
    if task=="Classification":
        CF_model=st.radio("Choose model",["KNN","DecisionTree","RandomForest","SVM"])
        
        if CF_model=="KNN":
            k=st.radio("neighbors",[3,5,7])
            
    
    output=st.selectbox("Select output data",df.columns)
    input=st.multiselect("Select input data",df.columns.drop(output))
    if len(input)==0:
        st.warning("Choose input value")
        st.stop()
    
    x=df[input]
    y=df[output]    
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)
    
    if st.button("Train model"):
        progress =st.progress(0)
        with st.spinner("Processing..."):
            for i in range(100):
                time.sleep(0.02)
                progress.progress(i+1)
                
        if task=="Regression":
            if RG_model == "LinearRegression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
            elif RG_model=="Polynomial":
                poly= PolynomialFeatures(degree=d)
                x_poly_train = poly.fit_transform(X_train)
                x_poly_test = poly.transform(X_test)
                model = LinearRegression()
                model.fit(x_poly_train, y_train)
                pred = model.predict(x_poly_test)
                
            st.write("R2 score:", r2_score(y_test, pred))
            st.write("Mean square error",mean_squared_error(y_test,pred))
                
            if len(input)>=1:
                feature_for_plot = input[0] 
                fig,data= plt.subplots()
                sns.scatterplot(x=df[feature_for_plot], y=df[output], data=df)
                data.set_xlabel(input)
                data.set_ylabel(output)
                st.pyplot(fig)
                    
        if task=="Classification":
            if CF_model=="KNN":
                model=KNeighborsClassifier(n_neighbors=k)

            elif CF_model=="DecisionTree":
                model=DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=42)
            
            elif CF_model=="RandomForest":
                model=RandomForestClassifier(n_estimators=10,criterion="entropy",max_depth=3,random_state=42)
                
            elif CF_model=="SVM":
                model=SVC(C=3,kernel="rbf",random_state=42)
                
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            st.write("Accuracy", accuracy_score(y_test, pred))
            z=confusion_matrix(y_test,pred)
            st.write("Classificatin report",classification_report(y_test,pred))
            
            if len(input)>=1:
                fig, ax = plt.subplots()
                sns.heatmap(z, annot=True, fmt="d", ax=ax)
                st.pyplot(fig)