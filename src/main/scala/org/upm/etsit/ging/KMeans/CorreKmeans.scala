package KMeans

import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler, StringIndexer, StandardScaler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random

/**
  * Created by Administrator on 11/01/2017.
  */

class CorreKmeans(private val spark: SparkSession) {

  // Cuenta clusteres

  def cuentaClusteres(data: DataFrame): Unit = {

    data.select("etiqueta").groupBy("etiqueta").count().orderBy($"count".desc).show(25)

    val numericOnly = data.drop("protocolo", "servicio", "flag").cache()

    val assembler = new VectorAssembler().
      setInputCols(numericOnly.columns.filter(_ != "etiqueta")).
      setOutputCol("vectorCaract")

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setPredictionCol("cluster").
      setFeaturesCol("vectorCaract")

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
    val pipelineModel = pipeline.fit(numericOnly)
    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

    kmeansModel.clusterCenters.foreach(println)

    val withCluster = pipelineModel.transform(numericOnly)

    withCluster.select("cluster", "etiqueta").
      groupBy("cluster", "etiqueta").count().
      orderBy($"cluster", $"count".desc).
      show(25)

    numericOnly.unpersist()
  }

}

