package dist.recsys

import org.apache.spark.SparkConf
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._

import scala.util.Try

object FpGrowth {
  private type ITEM = String
  private type LIFT = Double

  private val ordering: Ordering[(ITEM, LIFT)] = new Ordering[(ITEM, LIFT)] {
    override def compare(self: (ITEM, LIFT), that: (ITEM, LIFT)): Int = {
      val (_, lift1) = self
      val (_, lift2) = that
      lift1.compareTo(lift2)
    }
  }

  def main(args: Array[String]): Unit = {
    val spark = this.basicSpark
    val minSupport = Try(args(0).toInt).getOrElse(3)
    val minConfidence = Try(args(1).toDouble).getOrElse(0.1)
    val associationCounts = Try(args(2).toInt).getOrElse(10)
    val partitions = Try(args(3).toInt).getOrElse(2000)


    val transactions: RDD[Array[String]] = spark.sqlContext.sql(
      "select transactions from recsys.data_fpgrowth")
      .rdd
      .map(r => {
        /* transactions 以 空格分隔 的字符串 */
        val items = r.getAs[String]("transactions").split("\\s+")
        items
      })

    val i1i2Association = fpgRunner(
      spark,
      transactions,
      minSupport,
      minConfidence,
      associationCounts,
      partitions)

    i1i2Association
      .write
      .mode(SaveMode.Overwrite)
      .saveAsTable("recsys.model_fpgrowth")

    spark.stop()
  }

  private def fpgRunner(spark: SparkSession,
                        fpgData: RDD[Array[String]],
                        minSupportCount: Int,
                        minConfidence: Double,
                        associationCounts: Int,
                        partitions: Int) = {

    import spark.implicits._
    fpgData.checkpoint()
    val transactionsCount: Long = fpgData.count
    val fpg = new FPGrowth()
      .setMinSupport(minSupportCount * 1.0 / transactionsCount)
      .setNumPartitions(spark.sparkContext.defaultParallelism)

    val model = fpg.run(fpgData)
    val itemFrequents = model.freqItemsets.filter(_.items.length == 1)
      .map(itemSet => (itemSet.items.head, itemSet.freq))
      .collectAsMap()

    val itemFrequentsBC = spark.sparkContext.broadcast(itemFrequents)

    model.generateAssociationRules(minConfidence).filter(
      rule =>
        rule.antecedent.length == 1 &&
          rule.consequent.length == 1 &&
          itemFrequentsBC.value.contains(rule.consequent.head))
      .map(rule => {
        val A = rule.antecedent.head
        val B = rule.consequent.head
        val confidenceA2B = rule.confidence
        val supportB = itemFrequentsBC.value(B).toDouble / transactionsCount
        val lift = confidenceA2B / supportB
        (A, (B, lift))
      }).topByKey(associationCounts)(ordering)
      .toDF("item1", "item2", "lift")
  }

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
}
