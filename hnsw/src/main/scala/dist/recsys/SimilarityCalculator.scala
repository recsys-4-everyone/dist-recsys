package dist.recsys

import com.github.jelmerk.spark.knn.hnsw.HnswSimilarity
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object SimilarityCalculator {
  def main(args: Array[String]): Unit = {
    val spark = this.spark

    val candidates = spark.createDataFrame(Seq(
      ("1000000", Vectors.dense(0.0110, 0.2341)),
      ("2000000", Vectors.dense(0.2300, 0.3891)),
      ("3000000", Vectors.dense(0.4300, 0.9891))
    )).toDF("id", "vector")

    val hnsw = new HnswSimilarity()
      .setIdentifierCol("id")
      .setQueryIdentifierCol("id")
      .setFeaturesCol("vector")
      .setNumPartitions(5) // spark 分区
      .setNumReplicas(3) // hnsw 索引复制的个数，越大则每秒能处理的 query 越多，消耗资源也就越大
      .setK(10) // 最近邻个数
      .setExcludeSelf(true)
      .setOutputFormat("full")

    val model = hnsw.fit(candidates).setPredictionCol("neighbors").setEf(10)

    val neighbors = model.transform(candidates)

    /**
      * id 该 id 对应的向量      邻居节点[邻居节点 id, 邻居与查询节点的距离, ...]
      * +-------+--------------+-----------------------------------------------------------------+
      * |id     |vector        |neighbors                                                        |
      * +-------+--------------+-----------------------------------------------------------------+
      * |1000000|[0.011,0.2341]|[[3000000, 0.0652126651848689], [2000000, 0.11621311928381084]]  |
      * |3000000|[0.43,0.9891] |[[2000000, 0.007649116698351999], [1000000, 0.0652126651848689]] |
      * |2000000|[0.23,0.3891] |[[3000000, 0.007649116698351999], [1000000, 0.11621311928381084]]|
      * +-------+--------------+-----------------------------------------------------------------+
      */
    neighbors.show(10, truncate = false)

    spark.stop()
  }

  private def spark: SparkSession =
    SparkSession
      .builder
      .config(sparkConf)
      .enableHiveSupport()
      .getOrCreate()

  private def sparkConf: SparkConf = {
    val conf = new SparkConf()
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.network.timeout", "600")
      .set("spark.sql.crossJoin.enabled", "true")
      .set("spark.driver.maxResultSize", "5g")
      .set("spark.rpc.message.maxSize", "1000")
  }
}
