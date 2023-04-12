# import libraries-----------
import math
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

    def connection(self):
        username = 'root'
        password = '123'
        host = 'localhost:3306'
        database = 'house'
        self.engine = create_engine("mysql+pymysql://" + username + ":" + password + "@" + host + "/" + database)
        print(self.engine.connect())

    # def display_all_data(self):
    #     desired_width = 320
    #     pd.set_option('display.width', desired_width)
    #     pd.set_option('display.max_columns', 21)

    def data_load_to_database(self):
        self.data = pd.read_csv("melb_data.csv")
        print("Data load to Database------------\n ")
        with self.engine.begin() as conn:
            self.data.to_sql(con=conn, name='melb_data', if_exists='replace', index=False)
            print("Data loading successfully----------\n")
            print(self.data)

    def print_data(self):
        print("Data printing started----------\n ")
        with self.engine.begin() as conn:
            query = text("SELECT * FROM melb_data")
            self.data = pd.read_sql_query(query, conn)
        print("Data Print Completed----------\n")
        return self.data

    def check_data(self):
        self.data.info()
        self.data.describe()
        self.data.isnull().sum()
        self.data['Car'].fillna(self.data['Car'].mean(), inplace=True)
        self.data['BuildingArea'].fillna(self.data['BuildingArea'].mean(), inplace=True)
        self.data['YearBuilt'].fillna(self.data['YearBuilt'].mode()[0], inplace=True)
        self.data['CouncilArea'].fillna(self.data['CouncilArea'].mode()[0], inplace=True)
        self.data.drop(columns='Address', inplace=True)
        self.data.nunique()

    # Apply Model------
    def encode(self, c):
        le = LabelEncoder()
        le.fit(self.data[c])
        self.data[c] = le.fit_transform(self.data[c])

    def fit_data(self):
        cols = ["Suburb", "Type", "Method", "SellerG", "CouncilArea", "Regionname", "Date"]
        for c in cols:
            self.encode(c)
        print(self.data.head())

    # Split data in X and y-----------------------
    def split_data(self):
        X = self.data.drop(columns='Price')
        print(X.head())
        y = self.data['Price'].copy()
        print(y.head())
        print(y.shape)
        print(X.shape)
        return X, y

    def train_test(self, X, y):
        xtrain, xtest = X[0:10000], X[10000:]
        ytrain, ytest = y[0:10000], y[10000:]
        print(xtrain.shape)
        print(ytrain.shape)
        print(xtest.shape)
        print(ytest.shape)
        return xtrain, ytrain, xtest, ytest

    def apply_model(self, xtrain, ytrain, xtest, ytest):
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

        return df1, df2, df3,pred1,pred2,pred3

    def store_result(self, df1, df2,df3,pred1,pred2,pred3):
        # Create New DataFrame including all models score---
        print("\n FINAL RESULT OF SCORE AND ERRORS-----\n ")
        report = pd.concat([df1, df2,df3], ignore_index=True)
        print(report)
        print("\n FINAL RESULT OF PREDICTION-----\n ")
        prediction_result = pd.concat([pred1,pred2,pred3], ignore_index=True)
        print(prediction_result)

        return report,prediction_result

    def export_data(self,report,prediction_result):

        with self.engine.begin() as conn:
            print("Data exporting-----\n")
            #conn.execute(text("truncate table report"))
            report.to_sql(con=conn, name='report', if_exists='replace', index=True)
            #conn.execute(text("truncate table prediction"))
            report,prediction_result.to_sql(con=conn, name='prediction', if_exists='replace', index=True)
            print(report,"\n")
            print(prediction_result,"\n")
            print("Data exporting successfully!!!!!!!!!!!!!!!!!!!!!!!!")

    def run(self):
        self.connection()
        #self.display_all_data()
        self.data_load_to_database()
        self.print_data()
        self.check_data()
        self.fit_data()

        X, y = self.split_data()
        print(X)
        print(y)
        xtrain, ytrain, xtest, ytest = self.train_test(X, y)
        print(xtrain)
        print(ytrain)
        print(xtest.shape)
        print(ytest.shape)
        df1, df2, df3,pred1,pred2,pred3 = self.apply_model(xtrain, ytrain, xtest, ytest)
        report,prediction_result = self.store_result(df1, df2, df3,pred1,pred2,pred3)
        self.export_data(report,prediction_result)

obj = House_Price_Predction()
obj.run()