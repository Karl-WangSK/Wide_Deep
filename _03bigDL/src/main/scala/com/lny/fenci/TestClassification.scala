package com.lny.fenci

import java.util

import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.text.{DistributedTextSet, TextSet}
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.zoo.pipeline.api.keras.metrics.Accuracy
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import org.apache.log4j.{Level => Level4j, Logger => Logger4j}
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser


case class TextClassificationParams(
                                     dataPath: String = "file:///D:/data/ml/aaa", embeddingPath: String = "D:\\data\\ml\\tencent3",
                                     classNum: Int = 2, tokenLength: Int = 200,
                                     sequenceLength: Int = 500, encoder: String = "cnn",
                                     encoderOutputDim: Int = 256, maxWordsNum: Int = 5000,
                                     trainingSplit: Double = 0.8, batchSize: Int = 256,
                                     nbEpoch: Int = 20, learningRate: Double = 0.01,
                                     partitionNum: Int = 4, model: Option[String] = None,
                                     outputPath: Option[String] = Some("data/ml/index3"))

object TextClassification {

  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level4j.INFO)

  def main(args: Array[String]): Unit = {
    val parser: OptionParser[TextClassificationParams] = new OptionParser[TextClassificationParams]("TextClassification Example") {}

    parser.parse(args, TextClassificationParams()).map { param =>
      val conf=new SparkConf()
        .setMaster("local[4]")
        .setAppName("aaa")
      val sc: SparkContext = NNContext.initNNContext(conf,"Text Classification Example")

      val spark = SparkSession.builder().getOrCreate()
      val sparksql: SQLContext = spark.sqlContext
      val textSet: DistributedTextSet = TextSet.readParquet("data/introduce/par",sparksql)
        .toDistributed(sc, param.partitionNum)
      println("Processing text dataset...")
      val transformed: TextSet = textSet.tokenize()
        .word2idx(removeTopN = 0, maxWordsNum = param.maxWordsNum)
        .shapeSequence(param.sequenceLength).generateSample()

      val Array(trainTextSet, valTextSet) = transformed.randomSplit(
          Array(param.trainingSplit, 1 - param.trainingSplit))


      val model = if (param.model.isDefined) {
        TextClassifier.loadModel(param.model.get)
      }
      else {
        val tokenLength = param.tokenLength
        require(tokenLength == 50 || tokenLength == 100 || tokenLength == 200 || tokenLength == 300,
          s"tokenLength for GloVe can only be 50, 100, 200, 300, but got $tokenLength")
        val wordIndex = transformed.getWordIndex
        println(wordIndex.toString())
        val gloveFile = param.embeddingPath + "/glove.6B.200d.txt"
        TextClassifier(param.classNum, gloveFile, wordIndex, param.sequenceLength,
          param.encoder, param.encoderOutputDim)
      }

      model.compile(
        optimizer = new Adagrad(learningRate = param.learningRate,
          learningRateDecay = 0.1),
        loss = SparseCategoricalCrossEntropy[Float](),
        metrics = List(new Accuracy()))
      model.fit(trainTextSet, batchSize = param.batchSize,
        nbEpoch = param.nbEpoch, validationData = valTextSet)

      val predictSet: TextSet = model.predict(valTextSet, batchPerThread = param.partitionNum)
      println("Probability distributions of the first five texts in the validation set:")
      predictSet.toDistributed().rdd.take(5).map(_.getPredict).foreach(println)
      if (param.outputPath.isDefined) {
        val outputPath = param.outputPath.get
        predictSet.toDistributed().rdd.map(_.getPredict.toTensor).saveAsTextFile(outputPath+"/predict.txt")
        model.saveModel(outputPath + "/text_classifier.model")
        transformed.saveWordIndex(outputPath + "/word_index.txt")
        println("Trained model and word dictionary saved")
      }
      sc.stop()
    }
  }
}
