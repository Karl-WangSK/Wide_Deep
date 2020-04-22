package com.lny.wide_n_deep

import com.lny.utils.TagUtils._
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adam, Top1Accuracy}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.examples.recommendation.WNDParams
import com.intel.analytics.zoo.models.recommendation._
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import com.lny.utils.{OSSUtils, TagUtils}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.{SparkConf, SparkContext}

object One688_Wide_n_deep {

  /**
    *
    * root
    * |-- company_name: string (nullable = true)
    * |-- company_status: string (nullable = true)
    * |-- score: integer (nullable = true)
    * |-- company_type: string (nullable = true)
    * |-- industry: string (nullable = true)
    * |-- registered_capital: string (nullable = true)
    * |-- registered_time: string (nullable = true)
    * |-- employe_scale: string (nullable = true)
    * |-- contributed_capital: string (nullable = true)
    * |-- province: string (nullable = true)
    * |-- city: string (nullable = true)
    * |-- 1688_chengxintong: integer (nullable = true)
    * |-- 1688_medal_of_trading: integer (nullable = true)
    * |-- 1688_business_model: string (nullable = true)
    * |-- 1688_sumSales: string (nullable = true)
    * |-- 1688phone_supply_alias: string (nullable = true)
    * |-- id: integer (nullable = true)
    * |-- label: integer (nullable = true)
    */
  def main(args: Array[String]): Unit = {
    //训练数据
    run(WNDParams("ml-1m",
      "wide_n_deep",
      "oss://LTAIQENhPMB5rMgl:AgCXsxreFgjXc12KhJtUtrRJNcV40T@dtbird-platform.oss-cn-shanghai-internal.aliyuncs.com/bigdl/",
      1024,
      10,
      None,
      "DRAM"))
  }
  // Company 样例类
  case class Company(company_name: String, status: String, score: Int, company_type: String, industry: String, registered_capital: String,
                     registered_time: String, employe_scale: String, contributed_capital: String, province: String, city: String
                     , chengxintong: Int, medal_of_trading: Int, business_model: String, sumSales: String, phone_supply_level: String, id: Int, label: Int,
                     introduce_label: String)


  def run(params: WNDParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf: SparkConf = new SparkConf().
      setAppName("WideAndDeepExample")
    val sc: SparkContext = NNContext.initNNContext(conf)
    val sqlContext: SQLContext = SQLContext.getOrCreate(sc)

    val (ratingDF, id, industry, province, city) =
      loadPublicData(sqlContext, params.inputDir)

    ratingDF.groupBy("label").count().show()
    val bucketSize = 100
    val localColumnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("status", "score", "chengxintong", "medal_of_trading",
        "phone_supply_level", "company_type", "registered_capital", "employe_scale", "contributed_capital", "business_model", "sumSales", "registered_time"),
      wideBaseDims = Array(10, 101, 13, 6, 16, 12, 11, 7, 11, 5, 11, 10),
      //      wideCrossCols = Array("score-money"),
      //      wideCrossDims = Array(bucketSize),
      //      indicatorCols = Array("score"),
      //      indicatorDims = Array(101),
      embedCols = Array("industry_id", "province_id", "city_id", "introduce_label"),
      embedInDims = Array(124, 34, 322, 3),
      embedOutDims = Array(64, 64, 64, 64)
      //      continuousCols = Array("score")
    )
    //create W_N_D Instance
    val wideAndDeep: WideAndDeep[Float] = WideAndDeep[Float](
      params.modelType,
      numClasses = 3,
      columnInfo = localColumnInfo)

    val isImplicit = false
    // assemble features
    val featureRdds: RDD[UserItemFeature[Float]] =
      assemblyFeature(isImplicit, ratingDF, localColumnInfo, params.modelType)
    //get train and test data
    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      featureRdds.randomSplit(Array(0.75, 0.25))
    val trainRdds: RDD[Sample[Float]] = trainpairFeatureRdds.map(x => x.sample)
    val validationRdds: RDD[Sample[Float]] = validationpairFeatureRdds.map(x => x.sample)
    //lr 、lrd
    val optimMethod = new Adam[Float](
      learningRate = 1e-3,
      learningRateDecay = 1e-4)

    wideAndDeep.compile(optimizer = optimMethod,
      loss = SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
      metrics = List(new Top1Accuracy[Float]())
    )
    //create  tensorboard
    wideAndDeep.setTensorBoard("/root/tianyanpro/train_logs", "training_wide_n_deep")
    //fit
    wideAndDeep.fit(trainRdds, batchSize = params.batchSize,
      nbEpoch = params.maxEpoch, validationData = validationRdds)
    //saveModel
    OSSUtils.deleteOssFile("bigdl/tagpredict/1688_data/model")
    wideAndDeep
      .saveModel(params.inputDir + "tagpredict/1688_data/model")
    //predict results
    val results: RDD[Activity] = wideAndDeep.predict(validationRdds)
    //    results.foreach(println)
    //predict class
    val resultsClass: RDD[Int] = wideAndDeep.predictClass(validationRdds)
    resultsClass.take(50).foreach(println)
    //user-item-pair  prediction
    val userItemPairPrediction: RDD[UserItemPrediction] = wideAndDeep.predictUserItemPair(validationpairFeatureRdds)
    userItemPairPrediction.take(50).foreach(println)
    //user Recs   &&  item Recs
    val userRecs: RDD[UserItemPrediction] = wideAndDeep.recommendForUser(validationpairFeatureRdds, 5)
    val itemRecs: RDD[UserItemPrediction] = wideAndDeep.recommendForItem(validationpairFeatureRdds, 5)
    userRecs.take(10).foreach(println)
    itemRecs.take(10).foreach(println)

    //LOSS
    //    import sqlContext.sparkSession.implicits._
    //    val trainloss: Dataset[(Long, Float, Double)] = sqlContext.createDataset(wideAndDeep.getTrainSummary("Loss"))
    //    val valideLoss: Dataset[(Long, Float, Double)] = sqlContext.createDataset(wideAndDeep.getValidationSummary("Loss"))
    //    val valideaccuracy: Dataset[(Long, Float, Double)] = sqlContext.createDataset(wideAndDeep.getValidationSummary("Top1Accuracy"))
    //    //训练 损失
    //    trainloss
    //      .coalesce(1)
    //      .write
    //      .mode(SaveMode.Overwrite)
    //      .csv(params.inputDir+"trainloss")
    //    //测试 损失
    //    valideLoss
    //      .coalesce(1)
    //      .write
    //      .mode(SaveMode.Overwrite)
    //      .csv(params.inputDir+"valideloss")
    //    //获取准确率
    //    valideaccuracy
    //      .coalesce(1)
    //      .write
    //      .mode(SaveMode.Overwrite)
    //      .csv(params.inputDir+"valideaccuracy")
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String): (DataFrame, Int, Int, Int, Int) = {
    import sqlContext.implicits._

    val ratingDF: DataFrame = sqlContext.read.parquet(dataPath + "tagpredict/1688_data/1688_sampleWithTextClassification")
      .rdd.map(x => {
      Company(x.getString(0), x.getString(1), x.getInt(2), x.getString(3), x.getString(4), x.getString(5), x.getString(6), x.getString(7)
        , x.getString(8), x.getString(9), x.getString(10), x.getInt(11), x.getInt(12), x.getString(13), x.getString(14), x.getString(15), x.getInt(16), x.getInt(17)
        , x.getString(18))
    }).toDF()

    val minMaxCompany: Row = ratingDF.agg(count("id").cast(IntegerType), countDistinct("industry").cast(IntegerType), countDistinct("province").cast(IntegerType)
      , countDistinct("city").cast(IntegerType)).collect()(0)
    val (id: Int, industry: Int, province: Int, city: Int) = (minMaxCompany.getInt(0), minMaxCompany.getInt(1), minMaxCompany.getInt(2), minMaxCompany.getInt(3))
    (ratingDF, id, industry, province, city)
  }

  // convert features to RDD[Sample[Float]]
  def assemblyFeature(isImplicit: Boolean = false,
                      ratingDF: DataFrame,
                      columnInfo: ColumnFeatureInfo,
                      modelType: String): RDD[UserItemFeature[Float]] = {
    //TODO：UDF
    // age and gender as cross features, gender its self as wide base features
    val bucketUDF = udf(Utils.buckBucket(100))
    //load UDF
    val moneyUDF = udf(TagUtils.moneyTransform())
    val chengxinUDF = udf(TagUtils.chengxintong())
    val medalUDF = udf(TagUtils.medal())
    val levelUDF = udf(TagUtils.supply_level())
    val statusUDF = udf(TagUtils.company_status)
    val typeUDF = udf(TagUtils.company_type)
    val registered_timeUDF = udf(TagUtils.registered_time)
    val employee_scaleUDF = udf(TagUtils.employee_scale)
    val bussiness_modelUDF = udf(TagUtils.bussiness_model)
    val sumsaleslUDF = udf(TagUtils.sumsales)
    val double2intUDF = udf(TagUtils.double2int)
    /**
      * case class Company(company_name: String, status: String, score: Int, company_type: String, industry: String, registered_capital: String,
      * registered_time: String, employe_scale: String, contributed_capital: String, province: String, city: String
      * , chengxintong: Int, medal_of_trading: Int, business_model: String, sumSales: String, phone_supply_level: String, id: Int, label: Int)
      */
    val userDfUse = ratingDF
      .withColumn("score-money", bucketUDF(col("score"), col("registered_capital")))
      .withColumn("registered_capital", moneyUDF(col("registered_capital")))
      .withColumn("contributed_capital", moneyUDF(col("contributed_capital")))
      .withColumn("chengxintong", chengxinUDF(col("chengxintong")))
      .withColumn("medal_of_trading", medalUDF(col("medal_of_trading")))
      .withColumn("phone_supply_level", levelUDF(col("phone_supply_level")))
      .withColumn("status", statusUDF(col("status")))
      .withColumn("company_type", typeUDF(col("company_type")))
      .withColumn("registered_time", registered_timeUDF(col("registered_time")))
      .withColumn("employe_scale", employee_scaleUDF(col("employe_scale")))
      .withColumn("business_model", bussiness_modelUDF(col("business_model")))
      .withColumn("sumSales", sumsaleslUDF(col("sumSales")))
      .withColumn("introduce_label", col("introduce_label").cast(IntegerType))

    //transform string2index
    val df1: DataFrame = industry_category(userDfUse)
    val df2: DataFrame = city_category(df1)
    val df3: DataFrame = province_category(df2)
      .withColumn("industry_id", double2intUDF(col("industry_id")))
      .withColumn("province_id", double2intUDF(col("province_id")))
      .withColumn("city_id", double2intUDF(col("city_id")))

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(ratingDF)
      negativeDF.unionAll(ratingDF.withColumn("label", lit(2)))
    }
    else df3
    // userId, itemId as embedding features
    val joined = unioned
      .select(col("id"), col("label"), col("score"), col("registered_capital"),
        col("chengxintong"), col("score-money"), col("medal_of_trading"), col("phone_supply_level"), col("status"), col("company_type")
        , col("registered_time"), col("employe_scale"), col("contributed_capital"), col("business_model"), col("sumSales"), col("city_id")
        , col("province_id"), col("industry_id"), col("introduce_label"))
    val rddOfSample: RDD[UserItemFeature[Float]] = joined.rdd.map(r => {
      val iid = r.getAs[Int]("id")
      UserItemFeature(1, iid, Utils.row2Sample(r, columnInfo, modelType))
    })
    rddOfSample
  }
}
