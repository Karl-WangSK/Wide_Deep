package logistic

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

object Logistic {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("Logistic Regression Test")
      .setMaster("local[2]")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    val data = MLUtils.loadLibSVMFile(sc, "data/aaa.txt")

    // Split data into training (75%) and test (25%).
    val splits = data.randomSplit(Array(0.75, 0.25), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(3)
      .run(training)

    // Compute raw scores on the test set.
    val predictionAndLabels: RDD[(Double, Double)] = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy

    println(s"Accuracy = $accuracy")

    // Save and load model
//    model.save(sc, "data/model")


    // 输出测试结果
    val showPrediction: Array[(Double, Double)] = predictionAndLabels.collect()

    for (i <- 0 to showPrediction.length - 1) {
      println(showPrediction(i)._1 + "\t" + showPrediction(i)._2)
    }

    // 计算误差并输出
    val metricss = new MulticlassMetrics(predictionAndLabels)
    val precision = metricss.precision
    println("Precision = " + precision)

    sc.stop()

  }
}
