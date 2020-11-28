package dist.recsys.metrics

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object Main {
  def basicSpark: SparkSession =
    SparkSession
      .builder
      .config(getSparkConf)
      .enableHiveSupport()
      .getOrCreate()

  def getSparkConf: SparkConf = {
    val conf = new SparkConf()
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.network.timeout", "600")
      .set("spark.sql.crossJoin.enabled", "true")
      .set("spark.driver.maxResultSize", "5g")
      .set("spark.rpc.message.maxSize", "1000")
  }

  def main(args: Array[String]): Unit = {
    val ks = args.map(_.toInt)
    val spark = basicSpark
    val test = spark.sqlContext.sql("select user, item, relevance, timestamp from recsys.user_behaviour")
    val i2i = spark.sqlContext.sql("select item1, item2, score from recsys.model")
    val metrics = Metrics(test, i2i).metricsAt(ks)
    println("**************************************")
    metrics.foreach { case (k, metric) => println(s"@$k $metric") }
    println("**************************************")
    spark.stop
  }
}
