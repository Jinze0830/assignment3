import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.feature.{IDF, HashingTF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
//import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
//import org.apache.spark.mllib.linalg.Vectors
/**
 * Created by hadoop on 10/6/15.
 */
object MlKMeans {
  val conf = new SparkConf().setAppName("KMeans").setMaster("local")
  val sc = new SparkContext(conf)
  val converageDist=1.0

  //centroid size

  def tfidf():RDD[Vector]={
    val lines=sc.textFile("/Users/hadoop/Desktop/docword.nips.txt")
    //val wordInfo=lines.map(s=>s.split(" ")).collect()
    val wordInfo=lines.filter(s=>s.split(" ").size==3).map(line=>(line.split(" ")(0)+" "+line.split(" ")(1)))
    val wordInfoSeq:RDD[Seq[String]]=wordInfo.map(_.split(" ").toSeq)
    wordInfoSeq.cache()
    val hashingTF = new HashingTF()
    val tf = hashingTF.transform(wordInfoSeq)
    tf.cache()
    val idf = new IDF().fit(tf)
    val tfIdf = idf.transform(tf)
    //tfIdf.foreach(v=>println(v))
    // val s:Array[Vector] =tfIdf.collect()
    return tfIdf

  }
  def test()={
  // Load and parse the data
    val data:RDD[Vector] = this.tfidf()
    //val parsedData = data.map(s => s.toDense).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 10
    val numIterations = 100
    val clusters = org.apache.spark.mllib.clustering.KMeans.train(data, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(data)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    // Save and load model
    clusters.save(sc, "myModelPath")
    val sameModel = org.apache.spark.mllib.clustering.KMeansModel.load(sc, "myModelPath")
  }

  def main(args: Array[String]) {
    this.test()
  }
}
