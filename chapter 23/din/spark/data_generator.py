# -*- coding: utf-8 -*-

# 这里因为要将数据保存在本地，所以 master 指定为 local, 同时指定 jars. 
# 启动命令: spark-submit data_generator.py --master local --jars spark-tensorflow-connector_2.11-1.15.0
from pyspark.sql.types import *
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('din_train_data').getOrCreate()

# 保存在本地，可以换成 HDFS、S3 等分布式存储路径
path = "file:///home/axing/din/dataset"

# 指定各特征类型
feature_names = [
                 StructField("label", LongType()),
                 StructField("user_id", StringType()),
                 StructField("age", IntegerType()),
                 StructField("gender", StringType()),
                 StructField("item_id", StringType()),
                 StructField("clicked_items_15d", ArrayType(StringType(), True))]

schema = StructType(feature_names)
test_rows = [
    [1, "user_id1", 22, "0", "item_id1", ["item_id2", "item_id3", "item_id4"]],
    [0, "user_id2", 33, "1", "item_id5", ["item_id6", "item_id7"]]
]
rdd = spark.sparkContext.parallelize(test_rows)
df = spark.createDataFrame(rdd, schema)

# 存储为 tfrecord 文件格式，文件内部的数据格式为 Example
df.write.format("tfrecords").option("recordType", "Example").save(path, mode="overwrite")

df = spark.read.format("tfrecords").option("recordType", "Example").load(path)
df.show()

# 打印 dataframe 结构
df.printSchema()
# root
#  |-- item_id: string (nullable = true)
#  |-- age: long (nullable = true)
#  |-- gender: string (nullable = true)
#  |-- clicked_items_15d: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- label: long (nullable = true)
#  |-- user_id: string (nullable = true)