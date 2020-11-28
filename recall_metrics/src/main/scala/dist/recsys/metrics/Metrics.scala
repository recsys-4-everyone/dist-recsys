package dist.recsys.metrics

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, LongType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import scala.collection.mutable.ArrayBuffer

class Metrics(groundTruth: Dataset[_], predicted: Dataset[_],
              userCol: String = "user", itemCol: String = "item", relevanceCol: String = "relevance", timestampCol: String = "timestamp",
              mainItemCol: String = "item1", similarItemCol: String = "item2", scoreCol: String = "score") extends Serializable {
  val spark: SparkSession = predicted.sparkSession

  type USER = String
  type ITEM = String
  type RELEVANCE = Double
  type SIM = Double
  type RECS = Array[String]

  import spark.implicits._

  private val labelAndPredictions: RDD[((ITEM, RELEVANCE), RECS)] = {
    import org.apache.spark.sql.functions._

    val g = groundTruth
    val p = predicted

    val user = col(userCol).as("user")
    val item = col(itemCol).as("item")
    val relevance = col(relevanceCol).cast(DoubleType).as("relevance")
    val timestamp = col(timestampCol).cast(LongType).as("timestamp")

    val mainItem = col(mainItemCol).as("item1")
    val similarItem = col(similarItemCol).as("item2")
    val score = col(scoreCol).cast(DoubleType).as("score")

    val left = g.select(user, item, relevance, timestamp).map {
      case Row(user: USER, item: ITEM, relevance: RELEVANCE, timestamp: Long) => (item, (user, relevance, timestamp))
    }

    val right = p.select(mainItem, similarItem, score).map {
      case Row(item1: ITEM, item2: ITEM, score: SIM) => (item1, (item2, score))
    }.rdd.aggregateByKey(ArrayBuffer[(ITEM, SIM)]())(
      (l, r) => {
        l.append(r)
        l
      },
      (l1, l2) => {
        l1.appendAll(l2)
        l1
      }
    )

    left.rdd.leftOuterJoin(right)
      .map { case (truth, ((u, rel, ts), recsOption)) => (u, ((truth, rel, ts), recsOption.getOrElse(ArrayBuffer())))
      }.aggregateByKey(ArrayBuffer[((ITEM, RELEVANCE, Long), ArrayBuffer[(ITEM, SIM)])]())(
      (l, r) => {
        l.append(r)
        l
      },
      (l1, l2) => {
        l1.appendAll(l2)
        l1
      }
    ).flatMap { case (_, truthAndRecs) =>
      val sorted = truthAndRecs.sortBy { case ((_, _, ts), _) => ts }
      val truths = sorted.tail.map { case ((truth, rel, _), _) => (truth, rel) }
      val recList = sorted.take(sorted.size - 1).map { case (_, recs) =>
        recs.toArray.sortBy { case (_, s) => s }.reverse.map { case (i, _) => i }
      }
      truths.zip(recList)
    }.cache
  }

  def metricsAt(ks: Array[Int]): Array[(Int, Metrics)] = {
    labelAndPredictions.flatMap { case ((truth, relevance), recs) =>
      for (k <- ks) yield {
        val topK = recs.take(k)
        val metrics = getMetrics(truth, relevance, topK)
        (k, metrics)
      }
    }.reduceByKey(mergeMetrics).collect()
      .map { case (k, metrics) =>
        (k, calculateSpecificMetrics(metrics))
      }
      .sortBy { case (k, _) => k }
  }

  private def calculateSpecificMetrics(metrics: MetricsStat) = {
    val precision = metrics.tp * 1.0 / metrics.tpfp
    val recall = metrics.tp * 1.0 / metrics.tpfn
    val f1 = 2 * precision * recall / (precision + recall + 1E-8)
    val mrr = metrics.mrr / metrics.tpfn
    val ndcg = metrics.ndcg / metrics.tpfn
    Metrics(precision, recall, f1, mrr, ndcg)
  }

  private def log2(x: Double) = math.log(x) / math.log(2)

  private def getMetrics(groundTruth: ITEM, relevance: RELEVANCE, recs: RECS): MetricsStat = {
    recs.indexOf(groundTruth) match {
      case pos if pos >= 0 =>
        val dcg = (math.pow(2, relevance) - 1) / log2(pos + 1 + 1)
        val iDCG = (math.pow(2, relevance) - 1) / log2(1 + 1) // groundTruth 必须在第一位
        val ndcg = dcg / iDCG
        MetricsStat(ndcg, 1 * 1.0 / (pos + 1), 1, 1, recs.length)
      case _ => MetricsStat(0.0, 0.0, 0, 1, recs.length)
    }
  }

  private def mergeMetrics(metrics1: MetricsStat, metrics2: MetricsStat): MetricsStat = {
    MetricsStat(metrics1.ndcg + metrics2.ndcg,
      metrics1.mrr + metrics2.mrr,
      metrics1.tp + metrics2.tp,
      metrics1.tpfn + metrics2.tpfn,
      metrics1.tpfp + metrics2.tpfp)
  }

  case class Metrics(precision: Double, recall: Double, f1: Double, mrr: Double, ndcg: Double)

  case class MetricsStat(ndcg: Double, mrr: Double, tp: Int, tpfn: Int, tpfp: Int)

}

object Metrics {
  def apply(groundTruth: DataFrame, prediction: DataFrame,
            userCol: String = "user", itemCol: String = "item", relevanceCol: String = "relevance", timestampCol: String = "timestamp",
            mainItemCol: String = "item1", similarItemCol: String = "item2", scoreCol: String = "score"
           ): Metrics = new Metrics(groundTruth, prediction, userCol, itemCol, timestampCol,
    mainItemCol, similarItemCol, scoreCol)
}
