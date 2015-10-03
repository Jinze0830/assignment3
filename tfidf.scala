/**
 * Created by hadoop on 10/2/15.
 */

import org.apache.commons.lang.mutable.Mutable
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.Map
import scala.collection.parallel.mutable

class KMeans {
  val conf = new SparkConf().setAppName("KMeans")
  new SparkContext(conf)
  val sc = new SparkContext(conf)
  def transform[D <: String](dataset: RDD[D]): RDD[Vector] = {
    dataset.map(this.tf)
  }
  def tf(document: String): Vector= {

    //    val lines = sc.textFile("/Users/hadoop/Desktop/vocab.nips.txt")
    //    val lineLengths = lines.map(s => s.length)
    //    val totalLength = lineLengths.reduce((a, b) => a + b)

    val lines1 = sc.textFile(document)
    val sumInfo: Array[String] = lines1.collect()
    val wordInfo: Array[String] = sumInfo.slice(3, sumInfo.length)
    //var termFrequencies:Map[Int , Double] = Map()
    var Frequencies: Map[Int, Double] = Map()
    //m+=(1->1.0)

    //val termFrequencies = mutable.HashMap.empty[Int, Double]
    for (word <- wordInfo) {
      var eachWordInfo: Array[String] = word.split(" ")
      var wordId = eachWordInfo(1).toInt
      var wordCount = eachWordInfo(2).toDouble
      Frequencies += (wordId -> wordCount)
    }
    val totalLength = sumInfo(2).toInt

    val tf: Vector=Vectors.sparse(totalLength, Frequencies.toSeq)

    return tf
  }
  def tfidf(documents:RDD[String]): RDD[Vector]={
    val KMeans = new KMeans()
    val tf: RDD[Vector] = KMeans.transform(documents)
    tf.cache()
    val idf = new IDF().fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf)
  }

}
