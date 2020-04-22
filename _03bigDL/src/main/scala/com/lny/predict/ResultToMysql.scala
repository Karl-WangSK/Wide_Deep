package com.lny.predict

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{SaveMode, SparkSession}

object ResultToMysql {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("OssWc")
      .getOrCreate()

    import spark.implicits._
    import org.apache.spark.sql.functions._
    import scala.collection.JavaConverters._
    val df1 = spark.sparkContext.textFile("oss://LTAIQENhPMB5rMgl:AgCXsxreFgjXc12KhJtUtrRJNcV40T@dtbird-platform.oss-cn-shanghai-internal.aliyuncs.com/bigdl/Predict_result/")
    val df2 = spark.read.parquet("oss://LTAIQENhPMB5rMgl:AgCXsxreFgjXc12KhJtUtrRJNcV40T@dtbird-platform.oss-cn-shanghai-internal.aliyuncs.com/bigdl/tagpredict/1688_data/1688_clean/")
      .withColumnRenamed("id", "itemId")
    val predict = df1.map { data =>
      val strs: Array[String] = data.split(",")
      (strs(0).replace("UserItemPrediction(", ""), strs(1), strs(2), strs(3).replace(")", ""))
    }.filter(data => data._4.replace(")", "").toDouble > 0.6)
      .toDF("userId", "itemId", "predict", "possibility")

    val predictionAndLabels: RDD[(Double, Double)] = df2.join(predict, Seq("itemId"), "right_outer")
      .map { data =>
        val label = data.getAs[Int]("label")
        val predict = data.getAs[String]("predict")
        (predict.toDouble, label.toDouble)
      }.rdd


    val metrics1 = new BinaryClassificationMetrics(predictionAndLabels)
    val f1Score = metrics1.fMeasureByThreshold()
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    // Instantiate metrics object
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    val re = df2.join(predict, Seq("itemId"), "right_outer")
      .where("label=3 and (predict=2 or predict=1)")
      .select("company_name")
    val tianyan = spark.read.parquet("oss://LTAIQENhPMB5rMgl:AgCXsxreFgjXc12KhJtUtrRJNcV40T@dtbird-platform.oss-cn-shanghai.aliyuncs.com/dataPlatform/etl/dtbird/process/1688/1688AndTian_v1/")
    tianyan.join(re, Seq("company_name"), "inner")
      .withColumn("supplier", col("supplier").cast(StringType))
      .write
      .format("jdbc")
      .option("driver", "com.mysql.jdbc.Driver")
      .option("url", "jdbc:mysql://47.103.196.71:3306?ssl=false")
      .option("dbtable", "dtbird.1688predict")
      .option("user", "birder")
      .option("password", "um65@p")
      .option("fetchsize", "2")
      .mode(SaveMode.Overwrite)
      .save()
  }

}
