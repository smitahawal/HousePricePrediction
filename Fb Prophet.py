# Import Libraries
import mysql.connector
import datetime
import traceback
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sqlalchemy import create_engine
class FbProphet:
    def __init__(self):
        self.file_name = None
        self.data = None
        self.model = None
        self.m = None
        self.prediction = None
        self.df_cv = None
        self.createlogfile()
        self.engine = None
        self.fbprophet = None
        self.pred = None


    def createlogfile(self):
        try:
            fname = "fbprophet"
            time = datetime.datetime.now()
            tm = time.strftime("%m-%d-%Y-%H-%M-%S")
            self.file_name = fname+'_'+str(tm)+'txt'
            with open(self.file_name,'a') as f:
                f.write('INFO: Log File Created\n')
        except Exception:
            return False,None,'ERROR: Exception Occurred, Log File Not Created'+traceback.format_exc()
        return True, 'INFO: Log File Created Successfully\n'


    def create_connection(self):
        try:
            username = 'root'
            password = ""
            host = "localhost"
            database = 'fbprophet'

            self.engine = create_engine("mysql+pymysql://" + username + ":" + password + "@" + host + "/" + database)
            print(self.engine)
        except Exception:
            return False, None,'ERROR: Exception Occurred, Connection Not Created'+traceback.format_exc()
        return True, self.engine,"INFO: Connection Created Successfully\n"


    def Load(self):
        try:
            print('Data loading started')
            self.data = pd.read_sql_query("select * from fbprophetdata",self.engine)
            print('data loading completed')
            print(self.data)
        except Exception:
            return False,None,'ERROR: Exception Occurred,Data Not Loaded'+traceback.format_exc()
        return True, self.data,'INFO: Data Loaded Successfully\n'



    def Info(self):
        try:
            Information = self.data.info()
            print('Information: \n',Information)
        except Exception:
            return False,None,'ERROR: Exception Occurred,Information Not Display'+traceback.format_exc()
        return True, Information,'INFO: Information Display Successfully\n'


    def Desc(self):
        try:
            Describe = self.data.describe()
            print('Describe: \n',Describe)
        except Exception:
            return False,None,'ERROR: Exception Occurred,Data Not Described'+traceback.format_exc()
        return True, Describe,'INFO: Data Described Successfully\n'


    def CleanData(self):
        try:
            nullvalues = self.data.isnull().sum()
            print('NullValues: \n',nullvalues)
        except Exception:
            return False,None,'ERROR: Exception Occurred,Data Not Cleaned'+traceback.format_exc()
        return True, nullvalues,'INFO: Data Cleaned Successfully\n'

    def DateTime(self):
        try:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            print(self.data)
        except Exception:
            return False,None,'ERROR: Exception Occurred, Date Column Not Converted To DateTime'+traceback.format_exc()
        return True, self.data,'INFO: Date Column Converted To DateTime Successfully\n'

    def DropColumns(self):
        try:
            self.data = self.data[['Date','Close']]
            self.data = self.data.rename(columns = {'Date':'ds','Close':'y'})
            print(self.data)
        except Exception:
            return False,None,'ERROR: Exception Occurred, Columns Not Dropped'+traceback.format_exc()
        return True, self.data,'INFO: Columns Dropped Successfully\n'

    def FitModel(self):
        try:
            self.model = Prophet(interval_width=0.5, daily_seasonality=False)
            self.model.add_country_holidays(country_name='IND')
            self.model.fit(self.data)
            Holidays = self.model.train_holiday_names.to_list()
            print('Holidays:',Holidays)
        except Exception:
            return False,None,'ERROR: Exception Occurred, Model Not Fitted'+traceback.format_exc()
        return True, self.model,'INFO: Model Fitted Successfully\n'

    def Future(self):
        try:
            future = self.model.make_future_dataframe(periods=60)
            prophet_forcast = self.model.predict(self.data)
            self.model.plot_components(prophet_forcast)
        except Exception:
            return False,None,'ERROR: Exception Occurred, Future Data Not Done'+traceback.format_exc()
        return True, prophet_forcast,'INFO: Future Data Done Successfully\n'

    def DropData(self):
        try:
            self.data = self.data.drop(self.data.index[50:])
            print(self.data)
        except Exception:
            return False,None,'ERROR: Exception Occurred, Data Not Dropped'+traceback.format_exc()
        return True, self.data,'INFO: Data Dropped Successfully\n'

    def Model(self):
        try:
            self.m = Prophet(daily_seasonality=True)
            self.m.fit(self.data)

            components = self.m.component_modes
            print('components: ',components)
        except Exception:
            return False,None,'ERROR: Exception Occurred,Model Not Apply'+traceback.format_exc()
        return True, self.m,'INFO: Model Apply Successfully\n'

    def FutureDates(self):
        try:
            futuredates = self.m.make_future_dataframe(periods = 60)
            self.prediction = self.m.predict(futuredates)
            print('prediction: ',self.prediction)

            self.m.plot(self.prediction)
            plt.title("Prediction of the NSE-Tata-Global-Beverages-Limited stock price using the prophet")
            plt.xlabel('Date')
            plt.ylabel('Close Stock Price')
            plt.show()
        except Exception:
            return False,None,'ERROR: Exception Occurred,Future Data Not Done',traceback.format_exc()
        return True, self.prediction,'INFO: Future Data Done Successfully\n'


#from prophet.plot import plot_plotly
#plot_plotly(self.m,self.prediction)

    def CrossValidation(self):
        try:
            from prophet.diagnostics import cross_validation

            self.df_cv = cross_validation(self.m, horizon=60)
            print(self.df_cv)
        except Exception:
            return False,None,'ERROR: Exception Occurred,Cross Validation Not Done'+traceback.format_exc()
        return True, self.df_cv,'INFO: Cross Validation Done Successfully\n'


    def PerformanceMatrics(self):
        try:
            from prophet.diagnostics import performance_metrics
            df_performance = performance_metrics(self.df_cv)
            print(df_performance)
        except Exception:
            return False,None,'ERROR: Exception Occurred, Performance Matrix Not Done'+traceback.format_exc()
        return True, df_performance,'INFO: Performance Matrix Done Successfully\n'

    def Drop_Columns(self):
        try:
            self.pred = self.df_cv[['ds','y']]
            print('Prediction: \n',self.pred)
        except Exception:
            return False,None,'Exception Occurred,Columns Not Dropped'+traceback.format_exc()
        return True, self.pred,'INFO: Columns Dropped Successfully\n'

    def Export_Data(self):
        try:

            from sqlalchemy import create_engine
            my_conn = create_engine("mysql+mysqldb://root:@localhost/fbprophet")

            self.pred.to_sql(con=my_conn, name='prediction of fbprophet', if_exists='append', index=False)
        except Exception:
            return False,None,'Exception Occurred, Data Not Exported'+traceback.format_exc()
        return True, my_conn,'Data Exported successfully\n'

    def run(self):
        status = None
        status = self.createlogfile()
        if status:
            status, con, reason = self.create_connection()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.Load()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.Info()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.Desc()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.CleanData()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.DateTime()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.DropColumns()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.FitModel()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.Future()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.DropData()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.Model()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.FutureDates()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.CrossValidation()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.PerformanceMatrics()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.Drop_Columns()
            with open(self.file_name, 'a') as f:
                f.write(reason)

        if status:
            status, con, reason = self.Export_Data()
            with open(self.file_name, 'a') as f:
                f.write(reason)



obj = FbProphet()
obj.run()