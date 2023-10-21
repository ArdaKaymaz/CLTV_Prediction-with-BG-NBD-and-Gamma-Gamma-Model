##############################################################
# CLTV Prediction with BG-NBD ve Gamma-Gamma
##############################################################

# The data set consists of information obtained from the past shopping behavior of customers who made their last purchases
# via OmniChannel (both online and offline shopping) in 2020 - 2021. #
# 12 variables, 19.945 observations

# master_id: Unique customer id
# order_channel: Channel of the shopping platform where the shopping was made (Android, iOS, Desktop, Mobile)
# last_order_channel: Channel where the last shopping was made.
# first_order_date: The date of the customer's initial purchase.
# last_order_date: The date of the customer's most recent purchase.
# last_order_date_online: The date of the customer's most recent online purchase.
# last_order_date_offline: The date of the customer's most recent offline purchase.
# order_num_total_ever_online: The total number of purchases the customer has made on online platforms.
# order_num_total_ever_offline: The total number of purchases the customer has made on offline platforms.
# customer_value_total_ever_offline: The total amount the customer has spent on offline purchases.
# customer_value_total_ever_online: The total amount the customer has spent on online purchases.
# interested_in_categories_12: The list of categories in which the customer has made purchases in the last 12 months.


###############################################################
# TASK 1: Data Preparation
###############################################################


import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

# 1. OmniChannel.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("data_20k.csv")
df = df_.copy()

# 2. Define the functions 'outlier_thresholds' and 'replace_with_thresholds' that are necessary to suppress outliers.
# Note: When calculating CLTV, the 'frequency' values should be integers. Therefore, round the lower and upper limits with 'round()'.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)


# 3. If there are outliers in the variables 'order_num_total_ever_online,'
# 'order_num_total_ever_offline,' 'customer_value_total_ever_offline,'
# and 'customer_value_total_ever_online,' suppress them. #

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)


# 4. Omnichannel customers indicate that they shop from both online and offline platforms.
# Create new variables for each customer to represent the total number of purchases and total spending.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Examine variable types. Convert the type of variables that represent dates to "datetime".

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

###############################################################
# TASK 2: Creating the CLTV Data Structure
###############################################################

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)


# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()

###############################################################
# TASK 3: Building the BG/NBD and Gamma-Gamma Models and Calculating the 6-Month CLTV
###############################################################

# 1. BG/NBD modelini kurunuz.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# Predict the expected purchases from customers within 3 months and add them to the CLTV dataframe as 'exp_sales_3_month.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# Predict the expected purchases from customers within 6 months and add them to the CLTV dataframe as 'exp_sales_6_month.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])


# 2. Fit the Gamma-Gamma model. Predict the mean value of customers and add them to the CLTV dataframe as exp_average_value.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()

# 3. Calculate the 6-month CLTV and add it to the dataframe with the name 'cltv'.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

###############################################################
# TASK 4: Creating Segments Based on CLTV
###############################################################

# 1. Segment all your customers into 4 groups (segments) based on the 6-month standardized CLTV and add group names to the dataset.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

###############################################################
# BONUS: Functionize the entire process.
###############################################################

def create_cltv_df(dataframe):

    # Data Preparation
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # Creating the CLTV data structure
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).dt.days) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).dt.days) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # # Gamma-Gamma Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # CLTV Prediction
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # CLTV Segmenting
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)
cltv_df.head(10)


