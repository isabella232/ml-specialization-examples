#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer


spark = SparkSession.builder.master("local").appName("black_friday").getOrCreate()
df = spark.read.option("header", "true").csv("gs://doitintl_black_friday/data/BlackFriday.csv")
df.printSchema()

categorical_cols = ['Product_ID',
 'Gender',
 'Age',
 'Occupation',
 'City_Category',
 'Stay_In_Current_City_Years',
 'Marital_Status',
 'Product_Category_1',
 'Product_Category_2',
 'Product_Category_3',]
target = 'Purchase'

minDf = df.withColumn('row_index', F.monotonically_increasing_id())

for column in categorical_cols:
    print('Transforming column: ', column)
    output_col =  "_" + column.lower()
    indexer = StringIndexer(inputCol=column , outputCol=output_col)
    indexed = indexer.setHandleInvalid("keep").fit(df).transform(df)
    maxDf = indexed.withColumn('row_index', F.monotonically_increasing_id()).select('row_index', output_col)
    minDf = minDf.join(maxDf, on=["row_index"]).sort("row_index")
    
final_columns = [] 
for column in categorical_cols:
    final_columns.append(F.col("_" + column.lower()).alias(column.lower()))

final_columns.append(target)
minDf = minDf.select(final_columns)
minDf.coalesce(1).write.csv("gs://doitintl_black_friday/data/train_data.csv", header=True, mode="overwrite")


