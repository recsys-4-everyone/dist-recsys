package org.apache.spark.w2v

import org.apache.spark.SparkConf
import org.apache.spark.ml.embedding.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

import scala.collection.mutable

object Word2Vec {
  private type ITEM = String
  private type SIM = Double
  private val topN = 100
  private val vectorSize = 100
  private val maxIter = 10
  private val negative = 5
  private val sample = 1E-3
  private val maxSentenceLength = 1000
  private val numPartitions = 200
  private val learningRate = 0.025
  private val windowSize = 5
  private val minCount = 5

  private val ordering: Ordering[(ITEM, SIM)] = new Ordering[(ITEM, SIM)] {
    override def compare(self: (ITEM, SIM), that: (ITEM, SIM)): Int = {
      val (_, sim1) = self
      val (_, sim2) = that
      sim1.compareTo(sim2)
    }
  }

  def main(args: Array[String]): Unit = {
    val spark = this.basicSpark
    import spark.implicits._

    val corpus = spark.sqlContext.sql("select user_id, item_ids from rec.data_w2v")
    val trainingData = corpus
      .map(r => r.getAs[String]("item_ids").split(","))
      .toDF("ids")

    //Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("ids")
      .setVectorSize(vectorSize)
      .setMaxIter(maxIter)
      .setNegative(negative)
      .setHS(0)
      .setSample(sample)
      .setMaxSentenceLength(maxSentenceLength)
      .setNumPartitions(numPartitions)
      .setStepSize(learningRate)
      .setWindowSize(windowSize)
      .setMinCount(minCount)

    val model = word2Vec.fit(trainingData)

    val i1i2Similarities = cosineSimilarity(spark, model)

    i1i2Similarities
      .write
      .mode(SaveMode.Overwrite)
      .saveAsTable("recsys.model_word2vec")

    spark.stop()
  }

  def cosineSimilarity(spark: SparkSession, model: Word2VecModel): DataFrame = {
    val w2v = model.getVectors.rdd
      .map(r => {
        val word = r.getAs[String]("word")
        val vector = r.getAs[org.apache.spark.ml.linalg.Vector]("vector").toDense

        (word, vector)
      }).collect()

    val broadcast = spark.sparkContext.broadcast(w2v)

    import spark.implicits._

    model.getVectors.rdd.map(r => {
      val word = r.getAs[String]("word")
      val vector = r.getAs[org.apache.spark.ml.linalg.Vector]("vector").toDense
      (word, vector)
    }).flatMap { case (word1, vec1) =>
      val heap = new mutable.PriorityQueue[(String, Double)]()(ordering) // 构造小顶堆
      broadcast.value.filter { case (word2, _) => word2 != word1 }
        .foreach { case (word2, vec2) =>
          val dotProduct = vec1.toArray.zip(vec2.toArray).map { case (v1, v2) => v1 * v2 }.sum
          val norms = Vectors.norm(vec1, 2) * Vectors.norm(vec2, 2)
          val thisSim = math.abs(dotProduct) / norms
          if (heap.size < topN) heap.enqueue((word2, thisSim))
          else {
            heap.head match {
              case (_, minSim) if minSim < thisSim =>
                heap.dequeue()
                heap.enqueue((word2, thisSim))
              case _ => // 啥也不做
            }
          }
        }

      heap.toArray.map { case (word2, sim) => (word1, word2, sim) }
    }.toDF("item1", "item2", "sim")
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
