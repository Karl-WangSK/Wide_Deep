package com.lny.demo

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adam, Top1Accuracy}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.examples.recommendation.WNDParams
import com.intel.analytics.zoo.models.recommendation._
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext, SaveMode}

object Runner {

  case class Company(companyId: Int, score: Int, age: Int, number: Int)

  case class Tag(tagId: Int, tag: String, field: String)

  case class Rating(companyId: Int, tagId: Int, label: Int)


  def run(params: WNDParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().
      setAppName("WideAndDeepExample")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (ratingsDF, companyDF, tagDF, companyCount, tagCount) =
      loadPublicData(sqlContext, params.inputDir)

    ratingsDF.groupBy("label").count().show()
    val bucketSize = 100
    val localColumnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("score", "age"),
      wideBaseDims = Array(21, 3),
      wideCrossCols = Array("age-score"),
      wideCrossDims = Array(bucketSize),
      indicatorCols = Array("field", "score"),
      indicatorDims = Array(19, 3),
      embedCols = Array("companyId", "tagId"),
      embedInDims = Array(companyCount, tagCount),
      embedOutDims = Array(64, 64),
      continuousCols = Array("score"))

    val wideAndDeep: WideAndDeep[Float] = WideAndDeep[Float](
      params.modelType,
      numClasses = 5,
      columnInfo = localColumnInfo)

    val isImplicit = false
    val featureRdds =
      assemblyFeature(isImplicit, ratingsDF, companyDF, tagDF, localColumnInfo, params.modelType)

    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      featureRdds.randomSplit(Array(0.8, 0.2))
    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)

    val optimMethod = new Adam[Float](
      learningRate = 1e-2,
      learningRateDecay = 1e-5)

    wideAndDeep.compile(optimizer = optimMethod,
      loss = SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
      metrics = List(new Top1Accuracy[Float]())
    )
    //create  tensorboard
    wideAndDeep.setTensorBoard("/root/tianyanpro/bigdl/log_dir", "training_wideanddeep")

    wideAndDeep.fit(trainRdds, batchSize = params.batchSize,
      nbEpoch = params.maxEpoch, validationData = validationRdds)


    //predict results
    val results = wideAndDeep.predict(validationRdds)
    results.take(5).foreach(println)

    val resultsClass = wideAndDeep.predictClass(validationRdds)
    resultsClass.take(5).foreach(println)

    val userItemPairPrediction = wideAndDeep.predictUserItemPair(validationpairFeatureRdds)
    userItemPairPrediction.take(50).foreach(println)

    val userRecs = wideAndDeep.recommendForUser(validationpairFeatureRdds, 5)
    val itemRecs = wideAndDeep.recommendForItem(validationpairFeatureRdds, 5)

    userRecs.take(10).foreach(println)
    itemRecs.take(10).foreach(println)

    //LOSS
    import sqlContext.sparkSession.implicits._
    val trainloss: Dataset[(Long, Float, Double)] = sqlContext.createDataset(wideAndDeep.getTrainSummary("Loss"))
    val valideLoss: Dataset[(Long, Float, Double)] = sqlContext.createDataset(wideAndDeep.getValidationSummary("Loss"))
    val valideaccuracy: Dataset[(Long, Float, Double)] = sqlContext.createDataset(wideAndDeep.getValidationSummary("Top1Accuracy"))
    trainloss
      .coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .csv("oss://dtbird-platform/bigdl/trainloss")
    valideLoss
    .coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .csv("oss://dtbird-platform/bigdl/valideloss")

    valideaccuracy
      .coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .csv("oss://dtbird-platform/bigdl/valideaccuracy")

  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String):
  (DataFrame, DataFrame, DataFrame, Int, Int) = {
    import sqlContext.implicits._
    val ratings = sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::").map(n => n.toInt)
        Rating(line(0), line(1), line(2))
      }).toDF()
    val companyDF = sqlContext.read.text(dataPath + "/company.dat").as[String]
      .map(x => {
        val line = x.split("::")
        Company(line(0).toInt, line(1).toInt, line(2).toInt, line(3).toInt)
      }).toDF()
    val tagDF = sqlContext.read.text(dataPath + "/tag.dat").as[String]
      .map(x => {
        val line = x.split("::")
        Tag(line(0).toInt, line(1), line(2))
      }).toDF()

    val minMaxRow = ratings.agg(max("companyId"), max("tagId")).collect()(0)
    val (companyCount, tagCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))

    (ratings, companyDF, tagDF, companyCount, tagCount)
  }

  // convert features to RDD[Sample[Float]]
  def assemblyFeature(isImplicit: Boolean = false,
                      ratingDF: DataFrame,
                      companyDF: DataFrame,
                      tagDF: DataFrame,
                      columnInfo: ColumnFeatureInfo,
                      modelType: String): RDD[UserItemFeature[Float]] = {

    // age and gender as cross features, gender its self as wide base features
    val bucketUDF = udf(Utils.buckBucket(100))
    val genresList = Array("保健", "服务", "IT")
    val genresUDF = udf(Utils.categoricalFromVocabList(genresList))

    val userDfUse = companyDF
      .withColumn("age-score", bucketUDF(col("age"), col("score")))
    // genres as indicator
    val itemDfUse = tagDF
      .withColumn("field", genresUDF(col("field")))

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(ratingDF)
      negativeDF.unionAll(ratingDF.withColumn("label", lit(2)))
    }
    else ratingDF
    // userId, itemId as embedding features
    val joined = unioned
      .join(itemDfUse, Array("tagId"))
      .join(userDfUse, Array("companyId"))
      .select(col("companyId"), col("tagId"), col("label"), col("score"), col("age"),
        col("field"), col("age-score"))
    joined.show(1000, false)
    val rddOfSample = joined.rdd.map(r => {
      val uid = r.getAs[Int]("companyId")
      val iid = r.getAs[Int]("tagId")
      UserItemFeature(uid, iid, Utils.row2Sample(r, columnInfo, modelType))
    })
    rddOfSample
  }
}
