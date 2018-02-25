% matplotlib
inline
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from itertools import chain
from matplotlib import cm
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Row, functions, Column
from pyspark.sql.types import *
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import datetime
import elevation_grid as eg

spark = SparkSession.builder.appName('weather').getOrCreate()

from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.ml import Pipeline, Estimator
from pyspark.ml.feature import SQLTransformer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.regression import (LinearRegression,
                                   GBTRegressor,
                                   RandomForestRegressor,
                                   DecisionTreeRegressor)

from weather_tools import *

schema = StructType([
    StructField('station', StringType(), False),
    StructField('date', DateType(), False),
    # StructField('dayofyear', IntegerType(), False),
    StructField('latitude', FloatType(), False),
    StructField('longitude', FloatType(), False),
    StructField('elevation', FloatType(), False),
    StructField('tmax', FloatType(), False)
])


def get_data(inputloc, tablename='data'):
    data = spark.read.csv(inputloc, schema=schema)
    data.createOrReplaceTempView(tablename)
    return data


def make_weather_trainers(trainRatio,
                          estimator_gridbuilders,
                          metricName=None):
    """Construct a list of TrainValidationSplit estimators for weather data
       where `estimator_gridbuilders` is a list of (Estimator, ParamGridBuilder) tuples
       and 0 < `trainRatio` <= 1 determines the fraction of rows used for training.
       The RegressionEvaluator will use a non-default `metricName`, if specified.
    """
    feature_cols = ['latitude', 'longitude', 'elevation', 'doy']
    column_names = dict(featuresCol="features",
                        labelCol="tmax",
                        predictionCol="tmax_pred")

    getDOY = SQLTransformer(
        statement="select latitude,longitude,elevation,CAST(date_format(date,'D') AS int) as doy,tmax from __THIS__ ")  # TODO: engineer a day of year feature 'doy' from schema

    feature_assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol=column_names["featuresCol"])
    ev = (RegressionEvaluator()
          .setLabelCol(column_names["labelCol"])
          .setPredictionCol(column_names["predictionCol"])
          )
    if metricName:
        ev = ev.setMetricName(metricName)
    tvs_list = []
    for est, pgb in estimator_gridbuilders:
        est = est.setParams(**column_names)

        pl = Pipeline(stages=[getDOY, feature_assembler, est])  # TODO: Construct a pipeline with estimator est

        paramGrid = pgb.build()
        tvs_list.append(TrainValidationSplit(estimator=pl,
                                             estimatorParamMaps=paramGrid,
                                             evaluator=ev,
                                             trainRatio=trainRatio))
    return tvs_list


def get_best_weather_model(data, data1, flag):
    if (flag == "b1"):
        train = data
        test = data1
    else:
        train, test = data.randomSplit([0.75, 0.25])

    train = train.cache()
    test = test.cache()

    # e.g., use print(LinearRegression().explainParams()) to see what can be tuned
    estimator_gridbuilders = [
        #         estimator_gridbuilder(
        #             LinearRegression(),
        #             dict(regParam=[.2],  # [0.1, 0.01]
        #                  elasticNetParam=[.8],
        #                  aggregationDepth=[5],# 0-L2, 1-L1
        #                  maxIter=[110]
        #                  )),
        estimator_gridbuilder(
            GBTRegressor(),
            dict(maxDepth=[12],  # 0-L2, 1-L1
                 maxIter=[20]))
        #         estimator_gridbuilder(
        #             RandomForestRegressor(),
        #         dict(featureSubsetStrategy=['onethird'])),
        #         estimator_gridbuilder(
        #             DecisionTreeRegressor(),
        # # dict(maxDepth=[10],  # [0.1, 0.01]
        #      minInstancesPerNode=[2],  # 0-L2, 1-L1
        #      minInfoGain=[0.5]))

        # TODO: find better estimators

    ]
    metricName = 'r2'
    tvs_list = make_weather_trainers(.2,  # fraction of data for training
                                     estimator_gridbuilders,
                                     metricName)
    ev = tvs_list[0].getEvaluator()
    scorescale = 1 if ev.isLargerBetter() else -1
    model_name_scores = []
    for tvs in tvs_list:
        model = tvs.fit(train)
        test_pred = model.transform(test)
        score = ev.evaluate(test_pred) * scorescale
        model_name_scores.append((model, get_estimator_name(tvs.getEstimator()), score))
    best_model, best_name, best_score = max(model_name_scores, key=lambda triplet: triplet[2])
    print(
        "Best model is %s with validation data %s score %f" % (best_name, ev.getMetricName(), best_score * scorescale))
    return best_model


def main_weather(data, data1, flag):
    model = get_best_weather_model(data, data1, flag)
    print("Best parameters on test data:\n", get_best_tvs_model_params(model))
    if (flag == "b1"):
        data_pred = model.transform(data1).drop("features")
    else:
        data_pred = model.transform(data).drop("features")
    return data_pred


def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    #     m.shadedrelief(scale=scale)
    m.drawcountries()
    m.drawcoastlines()
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)

    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')


schema = StructType([
    StructField('station', StringType()),
    StructField('date', DateType()),
    StructField('latitude', FloatType()),
    StructField('longitude', FloatType()),
    StructField('elevation', FloatType()),
    StructField('tmax', FloatType()),
])
data = spark.read.csv("tmax-2/", schema=schema)
df = data.toPandas()
df['Year'] = df['date'].apply(lambda x: x.year)


def createmap(df):
    lat = df['latitude'].values
    long = df['longitude'].values
    fig = plt.figure(figsize=(15, 15), edgecolor='w')
    temp = df['tmax'].values
    m = Basemap(projection='cyl', resolution='l',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180, )
    lons, lats = m(long, lat)
    scatters = m.scatter(lons, lats, marker='o', c=temp, zorder=5, cmap='coolwarm', edgecolor='black')
    plt.colorbar(scatters, shrink=0.4, pad=0.05)
    draw_map(m)


createmap(df[df.Year <= 2008])
createmap(df[df.Year > 2008])
listpred = []
for latitude in range(-90, 90):
    for longitude in range(-180, 180):
        row = {'station': 'x', 'date': datetime.date.today(), 'latitude': float(latitude), 'longitude': float(longitude) \
            , 'elevation': float(eg.get_elevation(latitude, longitude)) \
            , 'tmax': 1.0}
        listpred.append(row)

data_inter = spark.sparkContext.parallelize(listpred)
schema = StructType([
    StructField('station', StringType()),
    StructField('date', DateType()),
    StructField('latitude', FloatType()),
    StructField('longitude', FloatType()),
    StructField('elevation', FloatType()),
    StructField('tmax', FloatType()),
])
data1 = spark.createDataFrame(data_inter, schema)
pred = main_weather(data, data1, "b1")
df = pred.toPandas()


# df['Year']=df['Date'].apply(lambda x: x.year)
def createmap2(df):
    lat = df['latitude'].values
    long = df['longitude'].values
    fig = plt.figure(figsize=(15, 15), edgecolor='w')
    temp = df['tmax_pred'].values
    m = Basemap(projection='cyl', resolution='l',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180, )
    lons, lats = m(long, lat)
    scatters = m.scatter(lons, lats, marker='o', c=temp, cmap='coolwarm')
    plt.colorbar(scatters, shrink=0.4, pad=0.05)
    draw_map(m)


createmap2(df)
pred = main_weather(data, None, "b2")
df = pred.toPandas()
df['tdiff'] = df['tmax'] - df['tmax_pred']


def createmap3(df):
    lat = df['latitude'].values
    long = df['longitude'].values
    fig = plt.figure(figsize=(15, 15), edgecolor='w')
    temp = df['tdiff'].values
    m = Basemap(projection='cyl', resolution='l',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180, )
    lons, lats = m(long, lat)
    scatters = m.scatter(lons, lats, marker='o', c=temp, zorder=5, cmap='coolwarm', edgecolor='black')
    plt.colorbar(scatters, shrink=0.4, pad=0.05)
    draw_map(m)


createmap3(df)
