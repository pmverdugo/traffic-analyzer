package org.upm.etsit.ging.KNN

/**
  * Created by Administrator on 11/01/2017.
  */
class CorreKNN {

  // Detecta cifrado

  def DetectaCifrado(data: DataFrame): Unit = {
    val pipelineModel = encajaPipeline(data, 180)

    val kMeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    val centroids = kMeansModel.clusterCenters

    val clustered = pipelineModel.transform(data)
    val threshold = clustered.
      select("cluster", "scaledvectorCaract").as[(Int, Vector)].
      map { case (cluster, vec) => Vectors.sqdist(centroids(cluster), vec) }.
      orderBy($"value".desc).take(100).last

    val originalCols = data.columns
    val anomalies = clustered.filter { row =>
      val cluster = row.getAs[Int]("cluster")
      val vec = row.getAs[Vector]("scaledvectorCaract")
      Vectors.sqdist(centroids(cluster), vec) >= threshold
    }.select(originalCols.head, originalCols.tail: _*)

    println(anomalies.first())
  }
}
