package com.lny.dataPre.data_interface

import com.lny.utils.ForFastextData.{attach_label, extract_label}
import org.apache.spark.sql.SparkSession

object MergeFasttextLabel {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("datapre")
      .getOrCreate()

    try {
      extract_label(spark = spark, transaction = args(0))

      attach_label(spark = spark, transaction = args(0))

      spark.stop()
    } catch {
      case e: Exception =>
        e.printStackTrace()
        System.exit(0)
    }
  }
}
