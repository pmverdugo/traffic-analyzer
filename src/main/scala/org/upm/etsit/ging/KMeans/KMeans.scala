/**
  * Created by Administrator on 11/01/2017.
  */

package org.upm.etsit.ging.KMeans

import org.apache.spark.sql.SparkSession

object KMeans {

  def main(args: Array[String]): Unit = {

    // inicializar contexto spark
    val spark = SparkSession.builder().getOrCreate()

    // importar paquetes recibidos
    val data = spark.read.
      option("inferSchema", true).
      option("header", false).
      csv("hdfs:///trafico.data").
      toDF("duracion", "longitud", "protocolo", "servidor")

    data.cache()

    // deteccion de cifrado por KNN
    val correKMeans = new CorreKmeans(spark)
    correKMeans.cuentaClusteres(data)

    // clustering por costes
    val costes = new Costes(spark)
    costes.muestraResultados(data)

    // clustering por ohp
    val ohp = new OHP(spark)
    ohp.muestraResultados(data)

    // clustering por sfv
    val sfv = new SFV(spark)
    sfv.muestraResultados(data)

    // clustering por entropia
    val entr = new Entropia(spark)
    entr.muestraResultados(data)

    data.unpersist()
  }
}