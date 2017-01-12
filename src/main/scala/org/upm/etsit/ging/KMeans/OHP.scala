package org.upm.etsit.ging.KMeans

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random

/**
  * Created by Administrator on 11/01/2017.
  */
class OHP(private val spark: SparkSession) {
  // Calcula OHP

  def muestraResultados(data: DataFrame): Unit = {
    (60 to 270 by 30).map(k => (k, calculaOHP(data, k))).foreach(println)
  }

  def calculaOHP(data: DataFrame, k: Int): Double = {
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocolo")
    val (servicioEncoder, servicioVecCol) = oneHotPipeline("servicio")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // columnas originales / columnas codificadas por vector
    val assembleCols = Set(data.columns: _*) --
      Seq("etiqueta", "protocolo", "servicio", "flag") ++
      Seq(protoTypeVecCol, servicioVecCol, flagVecCol)
    val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("vectorCaract")

    val scaler = new StandardScaler()
      .setInputCol("vectorCaract")
      .setOutputCol("scaledvectorCaract")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledvectorCaract").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, servicioEncoder, flagEncoder, assembler, scaler, kmeans))
    val pipelineModel = pipeline.fit(data)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(pipelineModel.transform(data)) / data.count()
  }

  def oneHotPipeline(inputCol: String): (Pipeline, String) = {
    val indexer = new StringIndexer().
      setInputCol(inputCol).
      setOutputCol(inputCol + "_indexed")
    val encoder = new OneHotEncoder().
      setInputCol(inputCol + "_indexed").
      setOutputCol(inputCol + "_vec")
    val pipeline = new Pipeline().setStages(Array(indexer, encoder))
    (pipeline, inputCol + "_vec")
  }
}
