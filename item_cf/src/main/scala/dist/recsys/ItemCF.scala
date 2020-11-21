package dist.recsys

import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.countDistinct
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.sql.{Dataset, Row, SaveMode, SparkSession}

import scala.collection.mutable.ArrayBuffer

object ItemCF {
  private type USER = String
  private type ITEM = String
  private type RATING = Double
  private type SIM = Double
  private type SIMSUM = Double
  private type COUNT = Long

  private val N = 10
  private val UPPER_BOUND = 3000
  private val LOWER_BOUND = 2

  private val ordering: Ordering[(ITEM, SIM)] = new Ordering[(ITEM, SIM)] {
    override def compare(self: (ITEM, SIM), that: (ITEM, SIM)): Int = {
      val (_, sim1) = self
      val (_, sim2) = that
      sim1.compareTo(sim2)
    }
  }

  private case class UserItemRating(user: USER, item: ITEM, rating: RATING)

  def main(args: Array[String]): Unit = {
    val spark = this.spark
    import spark.implicits._

    val userItemRating: Dataset[(USER, (ITEM, RATING))] =
      spark.sqlContext.sql("select user, item, rating from recsys.data_itemcf")
        .map { case Row(user: USER, item: ITEM, rating: RATING) => (user, (item, rating)) }

    /* 过滤无效用户 */
    val filteredUsers = userItemRating
      .groupBy("user").agg(countDistinct("item") as "item_count")
      .filter($"item_count" <= UPPER_BOUND && $"item_count" >= LOWER_BOUND)
      .select("user").as[String].collect()

    val filteredUsersBC = spark.sparkContext.broadcast(filteredUsers)

    val filteredUserItemRating: Dataset[(USER, (ITEM, RATING))] = userItemRating.filter(!$"user".isin(filteredUsersBC.value: _*))

    val itemCounts = filteredUserItemRating.
      groupBy("item").agg(countDistinct("user") as "user_count")
      .select("item", "user_count")
      .map { case Row(item: ITEM, userCount: COUNT) => (item, userCount) }
      .rdd
      .collectAsMap()

    val itemCountBC: Broadcast[collection.Map[ITEM, COUNT]] = spark.sparkContext.broadcast(itemCounts)

    val userItemRatings: RDD[(USER, ArrayBuffer[(ITEM, RATING)])] =
      filteredUserItemRating
        .rdd.aggregateByKey(ArrayBuffer[(ITEM, RATING)]())(
        (partitionRecords: ArrayBuffer[(ITEM, RATING)], singleRecord: (ITEM, RATING)) => {
          partitionRecords.append(singleRecord)
          partitionRecords
        },
        (partition1Records: ArrayBuffer[(ITEM, RATING)], partition2Records: ArrayBuffer[(ITEM, RATING)]) => {
          partition1Records.appendAll(partition2Records)
          partition1Records
        }
      ).filter { case (_: USER, itemRatings: ArrayBuffer[(ITEM, RATING)]) => itemRatings.length >= 2 }


    val itemTopNSimilarity: RDD[(ITEM, Array[(ITEM, SIM)])] =
      similarity(itemCountBC, userItemRatings)
        .map { case (item1, item2, sim) => (item1, (item2, sim)) }
        .topByKey(N)(ordering)

    /* save to be used */
    itemTopNSimilarity
      .flatMap { case (item1, simItems) => simItems.map { case (item2, sim) => (item1, item2, sim) } }
      .toDF("item1", "item2", "sim")
      .write
      .mode(SaveMode.Overwrite)
      .saveAsTable("recsys.model_itemcf")

    spark.stop
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

  private def similarity(itemCountBC: Broadcast[collection.Map[ITEM, COUNT]],
                         userItemRatings: RDD[(USER, ArrayBuffer[(ITEM, RATING)])]): RDD[(ITEM, ITEM, SIM)] = {
    val itemSimFromOneUser: RDD[((ITEM, ITEM), SIM)] = userItemRatings.flatMap {
      case (_: USER, itemRatingCounts: ArrayBuffer[(ITEM, RATING)]) =>
        for (ArrayBuffer(itemRating1, itemRating2) <- itemRatingCounts.combinations(2)) yield {
          val (item1, rating1) = itemRating1
          val (item2, rating2) = itemRating2
          val itemPair = (item1, item2)
          /* 惩罚热门用户 */
          val localSimilarity = 1 / (1 + Math.abs(rating1 - rating2)) / math.log1p(itemRatingCounts.size)
          (itemPair, localSimilarity)
        }
    }

    val itemSimilarity: RDD[(ITEM, ITEM, SIM)] =
      itemSimFromOneUser.aggregateByKey((0.0, 0L))(
        (similarityAndCountSum: (SIMSUM, COUNT), similarity: SIM) => {
          val (similaritySum, countSum) = similarityAndCountSum
          (similaritySum + similarity, countSum + 1)
        },
        (similarityAndCountSum1: (SIMSUM, COUNT), similarityAndCountSum2: (SIMSUM, COUNT)) => {
          val (similaritySum1, countSum1) = similarityAndCountSum1
          val (similaritySum2, countSum2) = similarityAndCountSum2
          (similaritySum1 + similaritySum2, countSum1 + countSum2)
        }
      ).flatMap { case ((item1, item2), (similaritySum, _)) =>
        val count1 = itemCountBC.value(item1)
        val count2 = itemCountBC.value(item2)
        /* 惩罚热门物品 */
        val item12Sim = (item1, item2, similaritySum / math.sqrt(count1 * count2))
        val item21Sim = (item2, item1, similaritySum / math.sqrt(count1 * count2))
        Iterator(item12Sim, item21Sim)
      }

    itemSimilarity
  }
}
