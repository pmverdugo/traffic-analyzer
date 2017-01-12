package org.upm.etsit.ging.KMeans

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random

/**
  * Created by Administrator on 11/01/2017.
  */
class Costes(private val spark: SparkSession) {
  // calcula costes

  def muestraResultados(data: DataFrame): Unit = {
    val numericOnly = data.drop("protocolo", "servicio", "flag").cache()
    (20 to 100 by 20).map(k => (k, costes0(numericOnly, k))).foreach(println)
    (20 to 100 by 20).map(k => (k, costes1(numericOnly, k))).foreach(println)
    numericOnly.unpersist()
  }

  def costes0(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler().
      setInputCols(data.columns.filter(_ != "etiqueta")).
      setOutputCol("vectorCaract")

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("vectorCaract")

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

    val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
  }

  def costes1(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler().
      setInputCols(data.columns.filter(_ != "etiqueta")).
      setOutputCol("vectorCaract")

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("vectorCaract").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

    val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
  }
}
