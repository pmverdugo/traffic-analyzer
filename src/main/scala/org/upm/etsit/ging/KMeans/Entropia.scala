package org.upm.etsit.ging.KMeans

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random

/**
  * Created by Administrator on 11/01/2017.
  */

class Entropia(private val spark: SparkSession) {

  // calcula entropia

  def muestraResultados(data: DataFrame): Unit = {
    (60 to 270 by 30).map(k => (k, costeEntropia(data, k))).foreach(println)

    val pipelineModel = encajaPipeline(data, 180)
    val countByClusteretiqueta = pipelineModel.transform(data).
      select("cluster", "etiqueta").
      groupBy("cluster", "etiqueta").count().
      orderBy("cluster", "etiqueta")
    countByClusteretiqueta.show()
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

  def encajaPipeline(data: DataFrame, k: Int): PipelineModel = {
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocolo")
    val (servicioEncoder, servicioVecCol) = oneHotPipeline("servicio")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // columnas originales/ columnas codificadas en vector
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
    pipeline.fit(data)
  }

  def costeEntropia(data: DataFrame, k: Int): Double = {
    val pipelineModel = encajaPipeline(data, k)

    // predecir cluster por datos
    val clusteretiqueta = pipelineModel.transform(data).
      select("cluster", "etiqueta").as[(Int, String)]
    val weightedClusterEntropy = clusteretiqueta.
      // etiquetas por cluter
      groupByKey { case (cluster, _) => cluster }.
      mapGroups { case (_, clusteretiquetas) =>
        val etiquetas = clusteretiquetas.map { case (_, etiqueta) => etiqueta }.toSeq
        // Count etiquetas in collections
        val etiquetaCounts = etiquetas.groupBy(identity).values.map(_.size)
        etiquetas.size * entropia(etiquetaCounts)
      }.collect()

    // entropia media por tamaño de cluster
    weightedClusterEntropy.sum / data.count()
  }

  def entropia(counts: Iterable[Int]): Double = {
    val values = counts.filter(_ > 0)
    val n = values.map(_.toDouble).sum
    values.map { v =>
      val p = v / n
      -p * math.log(p)
    }.sum
  }
}
