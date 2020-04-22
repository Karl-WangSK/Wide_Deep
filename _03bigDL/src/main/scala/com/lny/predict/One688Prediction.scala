package com.lny.predict

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.examples.recommendation.WNDParams
import com.intel.analytics.zoo.models.recommendation._
import com.lny.utils.{OSSUtils, TagUtils}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.{SparkConf, SparkContext}
import com.lny.utils.TagUtils._

object One688Prediction {

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

    //load data
    val (ratingDF, id, industry, province, city) =
      loadPublicData(sqlContext, params.inputDir)
    println(industry, province, city)

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
      embedInDims = Array(industry, province, city, 3),
      embedOutDims = Array(64, 64, 64, 64)
      //      continuousCols = Array("score")
    )

    val isImplicit = false
    // assemble features
    val featureRdds: RDD[UserItemFeature[Float]] =
      assemblyFeature(isImplicit, ratingDF, localColumnInfo, params.modelType)

    val validationRdds: RDD[Sample[Float]] = featureRdds.map(x => x.sample)
    //load model
    val model: WideAndDeep[Float] = WideAndDeep
      .loadModel(params.inputDir + "model")

    //user-item-pair  prediction
    val userItemPairPrediction: RDD[UserItemPrediction] = model.predictUserItemPair(featureRdds)
    userItemPairPrediction.take(50).foreach(println)
    //predict result
    OSSUtils.deleteOssDir("bigdl/tagpredict/1688_data/Predict_result/")
    userItemPairPrediction
      .coalesce(1)
      .saveAsTextFile(params.inputDir + "tagpredict/1688_data/Predict_result")

    //user Recs   &&  item Recs
    val userRecs: RDD[UserItemPrediction] = model.recommendForUser(featureRdds, 5)
    val itemRecs: RDD[UserItemPrediction] = model.recommendForItem(featureRdds, 5)
    userRecs.take(10).foreach(println)
    itemRecs.take(10).foreach(println)
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String): (DataFrame, Int, Int, Int, Int) = {
    import sqlContext.implicits._

    val ratingDF: DataFrame = sqlContext.read.parquet(dataPath + "tagpredict/1688_data/1688_cleanWithTextClassification/")
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