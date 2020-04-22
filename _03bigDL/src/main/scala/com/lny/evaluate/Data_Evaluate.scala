package com.lny.evaluate

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Data_Evaluate {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("evaluate")
      .getOrCreate()
    import spark.implicits._
    try {

      val df1 = spark.sparkContext.textFile("oss://LTAIQENhPMB5rMgl:AgCXsxreFgjXc12KhJtUtrRJNcV40T@dtbird-platform.oss-cn-shanghai-internal.aliyuncs.com/bigdl/tagpredict/3000_data/Predict_result/")
      val df2 = spark.read.parquet("oss://LTAIQENhPMB5rMgl:AgCXsxreFgjXc12KhJtUtrRJNcV40T@dtbird-platform.oss-cn-shanghai-internal.aliyuncs.com/bigdl/tagpredict/3000_data/cleanWithTextClassification/")
        .withColumnRenamed("id", "itemId")
      val predict = df1.map { data =>
        val strs: Array[String] = data.split(",")
        (strs(0).replace("UserItemPrediction(", ""), strs(1), strs(2), strs(3).replace(")", ""))
      }
        .toDF("userId", "itemId", "predict", "possibility")

      predict.show(2)
      df2.show(2)

      val predictionAndLabels: RDD[(Double, Double)] = predict
        .map { data =>
          val label = 2
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


      spark.sparkContext.stop()
    }catch{
      case e:Exception=>e.printStackTrace()
        System.exit(0)
    }
  }

}
