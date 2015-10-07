//import breeze.linalg.sum
import org.apache.spark.mllib.linalg.{DenseVector, Vectors, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.feature. HashingTF
import org.apache.spark.mllib.feature.IDF
//import breeze.linalg.DenseVector
/**
 * Created by hadoop on 10/2/15.
 */
object KMeans {
  val conf = new SparkConf().setAppName("KMeans").setMaster("local")
  val sc = new SparkContext(conf)
  val converageDist=1.0

  //centroid size
  val K =10
//  implicit class VectorPublications(val vector : Vector) extends AnyVal {
//    def toBreeze : breeze.linalg.Vector[scala.Double] = vector.toBreeze
//
//  }
//  implicit class BreezeVectorPublications(val breezeVector : breeze.linalg.Vector[Double]) extends AnyVal {
//    def fromBreeze : Vector = breezeVector.fromBreeze
//  }
//  def addVector(av1: Array[Double], av2:Array[Double]): Array[Double] ={
//
//    val a=new Array[Double](av1.length)
////    val bv1 = new DenseVector(dv1.toArray)
////    val bv2 = new DenseVector(dv2.toArray)
//    for(i<-0 until av1.length){
//      a(i)= av1(i) + av2(i)
//    }
//    //val vectout=new DenseVector(a)
//    return a
//
//  }
//  def divVector(av1: Array[Double], d: Int): Array[Double] ={
//    //var dArray=new Array[Double](av1.size)
//    val asize=av1.size
//    val av=new Array[Double](av1.size)
//    for(i<-0 until asize){
//      av(i)=av1(i)/d
//    }
//
//
//    return av
//  }

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
  def KMeans() ={
    val tiData=this.tfidf()
    tiData.cache()
    // pick up k point as center
    var centroid:Array[Vector] =tiData.takeSample(false,K,50L)
    centroid.foreach{k=>println(k)
    }
    var temDist=1.0
    var loopTime=0
    //find closest central for points
    do {
      //loopTime+=1
      var closest = tiData.map(p => (closestPoint(p, centroid), p))

      var pointsGroup:RDD[(Int,Iterable[Vector])] = closest.groupByKey()

//      pointsGroup.foreach{K=>
//        print(K._2+", ")
//      }
      val newCentroids:Array[(Int,Vector)]= pointsGroup.mapValues(ps => average(ps.toSeq)).collect()

      var tempDist = 0.0

      for (i <- 0 until K) {
        tempDist += Vectors.sqdist(newCentroids(i)._2, centroid(i))
      }
      for (newPoint <- newCentroids) {
        centroid(newPoint._1) = newPoint._2
      }
    }while (temDist>converageDist|loopTime>=100)
//new converage point
    centroid.foreach{k=>println(k)
    }
  }

  def closestPoint(p: Vector, centers:Array[Vector]):Int={
    var bestRes =0
    var closest= Double.PositiveInfinity
    for (i<-0 until centers.length){

      val temDis = Vectors.sqdist(p,centers(i))
      if(temDis<closest){
        closest = temDis
        bestRes = i
      }
    }
    return bestRes
  }
  def average(ps: Seq[Vector]): Vector={
    val numVectors = ps.size
    var sum:Array[Double]=ps(0).toArray
    var out:Array[Double]=ps(0).toArray
    for(i<-1 until numVectors){
      //ps(i).toArray
      var px:Array[Double]=ps(i).toArray
      for(j<-0 until ps(0).size){
        sum(j)=sum(j)+px(j)
      }
    }
    for(i<-0 until sum.length){
      out(i)=sum(i)/numVectors
    }
    val vout=Vectors.dense(out)
    println(vout.toSparse)
    return vout
  }



  def main (args: Array[String]){
    this.KMeans()
  }

}

