
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

import xgboost as xgb
from xgboost import plot_importance

warnings.filterwarnings('ignore')

from typing import Tuple, Union, List

class DelayModel:
    
    @staticmethod
    def get_period_day(date):
        from datetime import datetime
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()

        if(date_time > morning_min and date_time < morning_max):
            return 'mañana'
        elif(date_time > afternoon_min and date_time < afternoon_max):
            return 'tarde'
        elif(
            (date_time > evening_min and date_time < evening_max) or
            (date_time > night_min and date_time < night_max)
        ):
            return 'noche'
        
    @staticmethod
    def is_high_season(fecha):
        from datetime import datetime
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)

        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0
        
    @staticmethod
    def get_min_diff(data):
        from datetime import datetime
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff
    
    @staticmethod
    def get_rate_from_column(data, column):
        delays = {}
        for _, row in data.iterrows():
            if row['delay'] == 1:
                if row[column] not in delays:
                    delays[row[column]] = 1
                else:
                    delays[row[column]] += 1
        total = data[column].value_counts().to_dict()

        rates = {}
        for name, total in total.items():
            if name in delays:
                rates[name] = round(total / delays[name], 2)
            else:
                rates[name] = 0

        return pd.DataFrame.from_dict(data = rates, orient = 'index', columns = ['Tasa (%)'])

    

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
        data['min_diff'] = data.apply(self.get_min_diff, axis = 1)
        
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        
        print(data.columns)
        
        training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111)
        
        features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
        pd.get_dummies(data['MES'], prefix = 'MES')], 
        axis = 1
        )
        target = data['delay']
        
        features, target = DelayModel.preprocess(data)
        x_train, x_test, y_train, y_test = DelayModel.split_data(features, target)
        
        
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        return

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        self._model.fit(features, target)
        
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:        
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        
        xgboost_y_preds = self._model.predict(features)
        xgboost_y_preds = [1 if y_pred > 0.5 else 0 for y_pred in xgboost_y_preds]
        
        return xgboost_y_preds
    
    def plot_flights_by_airline(self, data: pd.DataFrame) -> None:
        flights_by_airline = data['OPERA'].value_counts()
        plt.figure(figsize = (10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x = flights_by_airline.index, y = flights_by_airline.values, alpha=0.9)
        plt.title('Flights by Airline')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Airline', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()
    
    def plot_flights_by_day(self, data: pd.DataFrame) -> None:
        flights_by_day = data['DIA'].value_counts()
        plt.figure(figsize = (10, 2))
        sns.set(style = "darkgrid")
        sns.barplot(x = flights_by_day.index, y = flights_by_day.values, color = 'lightblue', alpha=0.8)
        plt.title('Flights by Day')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Day', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()
        
    def plot_flights_by_month(self, data: pd.DataFrame) -> None:
        flights_by_month = data['MES'].value_counts()
        plt.figure(figsize = (10, 2))
        sns.set(style = "darkgrid")
        sns.barplot(x = flights_by_month.index, y = flights_by_month.values, color = 'lightblue', alpha=0.8)
        plt.title('Flights by Month')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Month', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()
    
    def plot_flights_by_day_in_week(self, data: pd.DataFrame) -> None:
        flights_by_day_in_week = data['DIANOM'].value_counts()
        days = [
            flights_by_day_in_week.index[2], 
            flights_by_day_in_week.index[5], 
            flights_by_day_in_week.index[4], 
            flights_by_day_in_week.index[1], 
            flights_by_day_in_week.index[0], 
            flights_by_day_in_week.index[6], 
            flights_by_day_in_week.index[3]
        ]
        values_by_day = [
            flights_by_day_in_week.values[2], 
            flights_by_day_in_week.values[5], 
            flights_by_day_in_week.values[4], 
            flights_by_day_in_week.values[1], 
            flights_by_day_in_week.values[0], 
            flights_by_day_in_week.values[6], 
            flights_by_day_in_week.values[3]
        ]
        plt.figure(figsize = (10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x = days, y = values_by_day, color = 'lightblue', alpha=0.8)
        plt.title('Flights by Day in Week')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Day in Week', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()
        
    def plot_flights_by_type(self, data: pd.DataFrame) -> None:
        flights_by_type = data['TIPOVUELO'].value_counts()
        plt.figure(figsize = (10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x = flights_by_type.index, y = flights_by_type.values, alpha=0.9)
        plt.title('Flights by Type')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Type', fontsize=12)
        plt.show()
        
    def plot_flights_by_destination(self, data: pd.DataFrame) -> None:
        flight_by_destination = data['SIGLADES'].value_counts()
        plt.figure(figsize = (10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x = flight_by_destination.index, y = flight_by_destination.values, color = 'lightblue', alpha=0.8)
        plt.title('Flight by Destination')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Destination', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()
        
    #delay_rates = DelayModel.get_rate_from_column(data, 'column_name')
    
    def plot_delay_rate_by_destination(self, data: pd.DataFrame) -> None:
        destination_rate = self.get_rate_from_column(data, 'SIGLADES')
        destination_rate_values = data['SIGLADES'].value_counts().index
        plt.figure(figsize = (20,5))
        sns.set(style="darkgrid")
        sns.barplot(x = destination_rate_values, y = destination_rate['Tasa (%)'], alpha = 0.75)
        plt.title('Delay Rate by Destination')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Destination', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()
        
    def plot_delay_rate_by_month(self, data: pd.DataFrame) -> None:
        month_rate = self.get_rate_from_column(data, 'MES')
        month_rate_value = data['MES'].value_counts().index
        plt.figure(figsize = (20,5))
        sns.set(style="darkgrid")
        sns.barplot(x = month_rate_value, y = month_rate['Tasa (%)'], color = 'blue', alpha = 0.75)
        plt.title('Delay Rate by Month')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Month', fontsize=12)
        plt.xticks(rotation=90)
        plt.ylim(0,10)
        plt.show()
        
    def plot_delay_rate_by_day(self, data: pd.DataFrame) -> None:
        days_rate = self.get_rate_from_column(data, 'DIANOM')
        days_rate_value = data['DIANOM'].value_counts().index
        plt.figure(figsize = (20, 5))
        sns.set(style="darkgrid")
        sns.barplot(x = days_rate_value, y = days_rate['Tasa (%)'], color = 'blue', alpha = 0.75)
        plt.title('Delay Rate by Day')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Days', fontsize=12)
        plt.xticks(rotation=90)
        plt.ylim(0,7)
        plt.show()
        
    def plot_delay_rate_by_season(self, data: pd.DataFrame) -> None:
        high_season_rate = self.get_rate_from_column(data, 'high_season')
        plt.figure(figsize = (5, 2))
        sns.set(style="darkgrid")
        sns.barplot(x = ["no", "yes"], y = high_season_rate['Tasa (%)'])
        plt.title('Delay Rate by Season')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('High Season', fontsize=12)
        plt.xticks(rotation=90)
        plt.ylim(0,6)
        plt.show()
        
    def plot_delay_rate_by_flight_type(self, data: pd.DataFrame) -> None:
        flight_type_rate = self.get_rate_from_column(data, 'TIPOVUELO')
        flight_type_rate_values = data['TIPOVUELO'].value_counts().index
        plt.figure(figsize = (5, 2))
        sns.set(style="darkgrid")
        sns.barplot(x = flight_type_rate_values, y = flight_type_rate['Tasa (%)'])
        plt.title('Delay Rate by Flight Type')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Flight Type', fontsize=12)
        plt.ylim(0,7)
        plt.show()
        
    def plot_delay_rate_by_period_day(self, data: pd.DataFrame) -> None:
        period_day_rate = self.get_rate_from_column(data, 'period_day')
        period_day_rate_values = data['period_day'].value_counts().index
        plt.figure(figsize = (5, 2))
        sns.set(style="darkgrid")
        sns.barplot(x = period_day_rate_values, y = period_day_rate['Tasa (%)'])
        plt.title('Delay Rate by Period of Day')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Period', fontsize=12)
        plt.ylim(3,7)
        plt.show()
        
    def split_data(self, features: pd.DataFrame, target: pd.Series, test_size: float = 0.33, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = test_size, random_state = random_state)
        return x_train, x_test, y_train, y_test
        
    #features, target = DelayModel.preprocess(data)
    #x_train, x_test, y_train, y_test = DelayModel.split_data(features, target)
    #print(f"train shape: {x_train.shape} | test shape: {x_test.shape}")
    #y_train.value_counts('%')*100
    #y_test.value_counts('%')*100
    
    #after prediction
    #cm = confusion_matrix(y_test, xgboost_y_preds)
    #print(cm)
    #report = classification_report(y_test, xgboost_y_preds)
    #print(report)
    