# 这里因为要将数据保存在本地，所以 master 指定为 local, 同时指定 jars. 
# 启动命令: spark-submit --master local --jars spark-tensorflow-connector_2.11-1.15.0 data.py
from pyspark.sql.types import *
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('ltr_train_data').getOrCreate()

# 保存在本地，可以换成 HDFS、S3 等分布式存储路径
path = "file:///home/axing/ltr/dataset"

def get_training_data(records):
    """
    records: 同一个 pv_id 下的数据集合
    """
    relevances = [record[6] for record in records]
    # 如果此 pv_id 下全是曝光数据，直接返回
    if not any(relevances):
        return []
    # 用户基本特征在一个 pv_id 下是相同的，存储为单值
    user_id = records[0][1]
    age = records[0][2]
    gender = records[0][3]
    
    # 物品在同一个 pv_id 下聚合为列表/数组
    item_id = [record[4] for record in records]
    
    # 用户行为在同一个 pv_id 下聚合为数组的数组
    clicked_items_15d = [record[5] for record in records]
    
    row = [relevances]
    row.append(user_id)
    row.append(age)
    row.append(gender)
    row.append(item_id)
    row.append(clicked_items_15d)
    
    return row


# 指定各特征类型
feature_names = [
                 StructField("relevance", ArrayType(LongType())),
                 StructField("user_id", StringType()),
                 StructField("age", LongType()),
                 StructField("gender", StringType()),
                 StructField("item_id", ArrayType(StringType())),
                 StructField("clicked_items_15d", ArrayType(ArrayType(StringType())))
]

schema = StructType(feature_names)
test_rows = [
    ["pv123", "uid012", 18, "0", "item012", [], 1],
    ["pv123", "uid012", 18, "0", "item345", ["item012"], 0],
    ["pv456", "uid345", 25, "1", "item456", [], 2],
    ["pv456", "uid345", 25, "1", "item567", ["item456"], 1],
    ["pv456", "uid345", 25, "1", "item678", ["item456", "item567"], 1]
]
rdd = spark.sparkContext.parallelize(test_rows)
rdd = rdd.keyBy(lambda row: row[0]).groupByKey().mapValues(list)
rdd = rdd.map(lambda pv_id_and_records: get_training_data(pv_id_and_records[1]))
df = spark.createDataFrame(rdd, schema)

# 存储为 tfrecord 文件格式，文件内部的数据格式为 SequenceExample
df.write.format("tfrecords").option("recordType", "SequenceExample").save(path, mode="overwrite")

df = spark.read.format("tfrecords").option("recordType", "SequenceExample").load(path)
df.show()

# 打印 dataframe 结构
df.printSchema()
# root
#  |-- item_id: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- age: long (nullable = true)
#  |-- gender: string (nullable = true)
#  |-- clicked_items_15d: array (nullable = true)
#  |    |-- element: array (containsNull = true)
#  |    |    |-- element: string (containsNull = true)
#  |-- relevance: array (nullable = true)
#  |    |-- element: long (containsNull = true)
#  |-- user_id: string (nullable = true)