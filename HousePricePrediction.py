# import libraries-----------
import math
import datetime
import traceback
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine,text
import warnings
warnings.filterwarnings('ignore')

class House_Price_Predction:
    def __init__(self):
        self.data = None
        self.engine = None

    def createlogfile(self):
        try:
            fname = "house_Logfile"
            time = datetime.datetime.now()
            tm = time.strftime("%m-%d-%Y-%H-%M-%S")
            self.file_name = fname + '_' + str(tm) + '.txt'
            with open(self.file_name, 'a') as f:
                f.write('INFO : Log File Created\n')
        except Exception:
            return False,'ERROR: Exception Occurred, Log File Not Created\n'+traceback.format_exc()
        return True,'INFO: Log File Created Successfully\n'

    def connection(self):
        try:
            username = 'root'
            password = '123'
            host = 'localhost:3306'
            database = 'house'
            self.engine = create_engine("mysql+pymysql://" + username + ":" + password + "@" + host + "/" + database)
            print(self.engine.connect())
        except Exception:
            return False,'Error: Exception Occurred,Log File Not Created\n' + traceback.format_exc()
        return True,'INFO : Connection Created Successfully ! \n'

    # def display_all_data(self):
    #     desired_width = 320
    #     pd.set_option('display.width', desired_width)
    #     pd.set_option('display.max_columns', 21)

    def data_load_to_database(self):
        try:
            self.data = pd.read_csv("melb_data.csv")
            print("Data load to Database------------\n ")
            with self.engine.begin() as conn:
                self.data.to_sql(con=conn, name='melb_data', if_exists='replace', index=False)
                print("Data loading successfully----------\n")
                print(self.data)
        except Exception:
            return False,'Error:Data Not Loaded to Database \n' + traceback.format_exc()
        return True,'INFO : Data loading to Database Successfully ! \n'

    def print_data(self):
        try:
            print("Data printing started----------\n ")
            with self.engine.begin() as conn:
                query = text("SELECT * FROM melb_data")
                self.data = pd.read_sql_query(query, conn)
            print("Data Print Completed----------\n")
        except Exception:
            return False,None,'Error : Data Printing Error\n' + traceback.format_exc()
        return True,self.data,'INFO : Data Printing Successfully !\n'

    def check_data(self):
        try:
            self.data.info()
            self.data.describe()
            self.data.isnull().sum()
            self.data['Car'].fillna(self.data['Car'].mean(), inplace=True)
            self.data['BuildingArea'].fillna(self.data['BuildingArea'].mean(), inplace=True)
            self.data['YearBuilt'].fillna(self.data['YearBuilt'].mode()[0], inplace=True)
            self.data['CouncilArea'].fillna(self.data['CouncilArea'].mode()[0], inplace=True)
            self.data.drop(columns='Address', inplace=True)
            self.data.nunique()
        except Exception:
            return False,'Error : Data Checking Error\n' + traceback.format_exc()
        return True,'INFO : Data Check Successfully !\n'

    # Apply Model------
    def encode(self, c):
        le = LabelEncoder()
        le.fit(self.data[c])
        self.data[c] = le.fit_transform(self.data[c])

    def fit_data(self):
        try:
            cols = ["Suburb", "Type", "Method", "SellerG", "CouncilArea", "Regionname", "Date"]
            for c in cols:
                self.encode(c)
            print(self.data.head())
        except Exception:
            return False,'Error : Data Not Fit\n' + traceback.format_exc()
        return True,'INFO : Data Fit Successfully !\n'

    # Split data in X and y-----------------------
    def split_data(self):
        try:
            X = self.data.drop(columns='Price')
            print(X.head())
            y = self.data['Price'].copy()
            print(y.head())
            print(y.shape)
            print(X.shape)
            result = [X, y]
        except Exception:
            return False,'Error : Data Not Splitting Properly\n' + traceback.format_exc()
        return True,result,'INFO : Data Splitting Successfully !\n'

    def train_test(self, X, y):
        try:
            xtrain, xtest = X[0:10000], X[10000:]
            ytrain, ytest = y[0:10000], y[10000:]
            print(xtrain.shape)
            print(ytrain.shape)
            print(xtest.shape)
            print(ytest.shape)

            result = [xtrain, ytrain, xtest, ytest]
        except Exception:
            return False,'Error : Data Train and Test Error \n'+traceback.format_exc()
        return True,result,'INFO : Data Train and Test Successfully !\n'


    def apply_model(self, xtrain, ytrain, xtest, ytest):
        try:
            # Linear regression Model -------
            mlr_model = LinearRegression()
            mlr_model.fit(xtrain, ytrain)
            pred_mlr = mlr_model.predict(xtest)
            print(mlr_model.coef_)

            # LinearRegression Score and explained_variance_score
            mlr_score = mlr_model.score(xtest, ytest)
            expl_mlr = explained_variance_score(pred_mlr, ytest)

            # LinearRegression Errors--------
            lr_mae = mean_absolute_error(ytest, pred_mlr)
            lr_mse = mean_squared_error(ytest, pred_mlr)
            lr_mape = np.mean(np.abs((ytest - pred_mlr) / ytest)) * 100
            lr_msrt = math.sqrt(mean_squared_error(ytest, pred_mlr))
            print("--------------------",lr_mape)

            result = [['LinearRegression', mlr_score, expl_mlr, lr_mae, lr_mse, lr_mape, lr_msrt]]
            df1 = pd.DataFrame(result, columns=['Model', 'Score', 'ExplainedVarianceScore', 'Mae', 'Mse', 'Mape', 'Rmse'])

            # Apply  Decision Tree model ---------------------
            tr_regressor = DecisionTreeRegressor(random_state=0)
            tr_regressor.fit(xtrain, ytrain)
            tr_regressor.score(xtest, ytest)
            pred_tr = tr_regressor.predict(xtest)

            # DecisionTreeRegressor Score and explained_variance_score
            decision_score = tr_regressor.score(xtest, ytest)
            decision_expl_tr = explained_variance_score(pred_tr, ytest)

            # DecisionTreeRegressor Errors--------
            tr_mae = mean_absolute_error(ytest, pred_tr)
            tr_mse = mean_squared_error(ytest, pred_tr)
            tr_mape = np.mean(np.abs((ytest - pred_tr) / ytest)) * 100
            tr_msrt = math.sqrt(mean_squared_error(ytest, pred_tr))

            result = [['DecisionTreeRegressor', decision_score, decision_expl_tr, tr_mae, tr_mse, tr_mape, tr_msrt]]
            df2 = pd.DataFrame(result, columns=['Model', 'Score', 'ExplainedVarianceScore', 'Mae', 'Mse', 'Mape', 'Rmse'])

            # Random Forest Regression Model ---------------------
            random_forest = RandomForestRegressor(random_state=60)
            random_forest.fit(xtrain, ytrain)
            random_forest.score(xtest, ytest)
            pred_rf = random_forest.predict(xtest)

            # Random Forest Regression Score and explained_variance_score
            random_forest = tr_regressor.score(xtest, ytest)
            random_expl_scr = explained_variance_score(pred_rf, ytest)

            # Random Forest Regression Errors--------
            rf_mae = mean_absolute_error(ytest, pred_rf)
            rf_mse = mean_squared_error(ytest, pred_rf)
            rf_mape = np.mean(np.abs((ytest - pred_rf) / ytest)) * 100
            rf_msrt = math.sqrt(mean_squared_error(ytest, pred_rf))

            result = [['RandomForestRegression', random_forest, random_expl_scr, rf_mae, rf_mse, rf_mape, rf_msrt]]
            df3 = pd.DataFrame(result, columns=['Model', 'Score', 'ExplainedVarianceScore', 'Mae', 'Mse', 'Mape', 'Rmse'])

            pred_mlr = pd.DataFrame(pred_mlr)
            pred_mlr["Model"] = "LinearRegression"
            pred_mlr.rename(columns={0: 'Prediction'},inplace=True)
            pred1 = pd.DataFrame(pred_mlr)
            print("\n",pred1)

            pred_tr = pd.DataFrame(pred_tr)
            pred_tr["Model"] = "DecisionTreeRegressor"
            pred_tr.rename(columns={0: 'Prediction'}, inplace=True)
            pred2 = pd.DataFrame(pred_tr)
            print("\n",pred2)

            pred_rf = pd.DataFrame(pred_rf)
            pred_rf["Model"] = "RandomForestRegression"
            pred_rf.rename(columns={0: 'Prediction'}, inplace=True)
            pred3 = pd.DataFrame(pred_rf)
            print("\n", pred3)
            result = [df1, df2, df3,pred1,pred2,pred3]
        except Exception:
            return False,'ERROR : Apply Models Not Apply Successfully !\n'
        return True,result,'INFO : Models Apply Successfully !\n'

    def store_result(self, df1, df2,df3,pred1,pred2,pred3):
        try:
            # Create New DataFrame including all models score---
            print("\n FINAL RESULT OF SCORE AND ERRORS-----\n ")
            report = pd.concat([df1, df2,df3], ignore_index=True)
            print(report)
            print("\n FINAL RESULT OF PREDICTION-----\n ")
            prediction_result = pd.concat([pred1,pred2,pred3], ignore_index=True)
            print(prediction_result)
            result = [report,prediction_result]
        except Exception:
            return False,'ERROR : Result Could Not Stored-- \n'
        return True,result,'INFO : Result Stored Successfully!\n'

    def export_data(self,report,prediction_result):
        try:
            with self.engine.begin() as conn:
                print("Data exporting-----\n")
                #conn.execute(text("truncate table report"))
                report.to_sql(con=conn, name='report', if_exists='replace', index=True)
                #conn.execute(text("truncate table prediction"))
                report,prediction_result.to_sql(con=conn, name='prediction', if_exists='replace', index=True)
                print(report,"\n")
                print(prediction_result,"\n")
                print("Data exporting successfully!!!!!!!!!!!!!!!!!!!!!!!!")
        except Exception:
            return False,'ERROR : Data Not Exported--'
        return True,conn,'INFO : Data Exported Successfully!'

    def run(self):
        status = None
        status = self.createlogfile()
        if status:
            status,reason = self.connection()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status,reason = self.data_load_to_database()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status,result,reason = self.print_data()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status,reason = self.check_data()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        # self.encode(c)

        if status:
            status,reason = self.fit_data()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status,result,reason = self.split_data()
            X = result[0]
            y = result[1]
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status,result,reason = self.train_test(X, y)
            with open(self.file_name, 'a') as f:
                f.write(reason)
            xtrain = result[0]
            ytrain = result[1]
            xtest = result[2]
            ytest = result[3]

        if status:
            status,result,reason = self.apply_model(xtrain, ytrain, xtest, ytest)
            with open(self.file_name, 'a') as f:
                f.write(reason)
            df1 = result[0]
            df2 = result[1]
            df3 = result[2]
            pred1 = result[3]
            pred2 = result[4]
            pred3 = result[5]

        if status:
            status,result,reason = self.store_result(df1, df2, df3, pred1, pred2, pred3)
            with open(self.file_name, 'a') as f:
                f.write(reason)
            report = result[0]
            prediction_result = result[1]

        if status:
            status,result,reason = self.export_data(report, prediction_result)
            with open(self.file_name, 'a') as f:
                f.write(reason)

obj = House_Price_Predction()
obj.run()