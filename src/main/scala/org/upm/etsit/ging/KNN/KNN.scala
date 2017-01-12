package org.upm.etsit.ging.KNN

import org.apache.spark.sql.SparkSession

/**
  * Created by Administrator on 11/01/2017.
  */

object KNN {
  def main(args: Array[String]): Unit = {
    // inicializar contexto spark
    val spark = SparkSession.builder().getOrCreate()

    // importar paquetes raw
    val data = spark.read.
      option("inferSchema", true).
      option("header", false).
      csv("hdfs:///raw.tcpdump").
      toDF("duracion", "longitud", "protocolo", "servidor")

    data.cache()
    // deteccion de cifrado por KNN
    val correKNN = new CorreKNN(spark)
    correKNN.DetectaCifrado(data)
  }
}
