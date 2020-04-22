package com.lny.predict

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.{Adam, Top1Accuracy}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.examples.recommendation.WNDParams
import com.intel.analytics.zoo.models.recommendation._
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import com.lny.utils.{OSSUtils, TagUtils}
import com.lny.utils.TagUtils._
import com.lny.wide_n_deep.Kehu_Wide
import com.lny.wide_n_deep.Kehu_Wide.Company
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

object KehuPredict {
  def main(args: Array[String]): Unit = {
    //训练数据
    KehuPredict.run(WNDParams("ml-1m",
      "wide",
      "oss://LTAIQENhPMB5rMgl:AgCXsxreFgjXc12KhJtUtrRJNcV40T@dtbird-platform.oss-cn-shanghai-internal.aliyuncs.com/bigdl/",
      256,
      15,
      None,
      "DRAM"))
  }

  // Company 样例类
  case class Company(company_name: String, score: Int, status: String, company_type: String, industry: String, registered_capital: String,
                     registered_time: String, employe_scale: String, contributed_capital: String, vc_step: String, label: Int, icp_audit_time_format: String
                     , status_codes: String, id: Int, text_label: Int)


  def run(params: WNDParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf: SparkConf = new SparkConf().
      setAppName("WideAndDeepExample")
    val sc: SparkContext = NNContext.initNNContext(conf)
    val sqlContext: SQLContext = SQLContext.getOrCreate(sc)
    try {
      val (ratingDF, id, industry, step) =
        loadPublicData(sqlContext, params.inputDir)

      ratingDF.groupBy("label").count().show()
      val bucketSize = 100
      val localColumnInfo = ColumnFeatureInfo(
        wideBaseCols = Array("status", "score", "company_type", "registered_capital",
          "employe_scale", "contributed_capital", "registered_time", "industry_id", "step_id", "icp_audit_time_format",
          "status_pos", "status_neg", "text_label"),
        wideBaseDims = Array(10, 101, 9, 11, 7, 11, 10, 124, 22, 8, 8, 8, 3)
        //      wideCrossCols = Array("score-money"),
        //      wideCrossDims = Array(bucketSize),
        //      indicatorCols = Array("score"),
        //      indicatorDims = Array(101),
        //      embedCols = Array("industry_id", "province_id", "city_id", "introduce_label"),
        //      embedInDims = Array(124, 34, 322, 3),
        //      embedOutDims = Array(64, 64, 64, 64)
        //      continuousCols = Array("score")
      )

      val isImplicit = false
      // assemble features
      val featureRdds: RDD[UserItemFeature[Float]] =
        assemblyFeature(isImplicit, ratingDF, localColumnInfo, params.modelType)

      //get train and test data
      val validationRdds = featureRdds.map(x => x.sample)

      //load model
      val model: WideAndDeep[Float] = WideAndDeep
        .loadModel(params.inputDir + "tagpredict/3000_data/model")

      //create  tensorboard
      model.setTensorBoard("/root/tianyanpro/train_logs", "training_wide_n_deep")

      //user-item-pair  prediction
      val userItemPairPrediction: RDD[UserItemPrediction] = model.predictUserItemPair(featureRdds)
      userItemPairPrediction.take(50).foreach(println)
      //predict result
      OSSUtils.deleteOssDir("bigdl/tagpredict/3000_data/Predict_result/")
      userItemPairPrediction
        .coalesce(1)
        .saveAsTextFile(params.inputDir + "tagpredict/3000_data/Predict_result")
      //user Recs   &&  item Recs
      val userRecs: RDD[UserItemPrediction] = model.recommendForUser(featureRdds, 5)
      val itemRecs: RDD[UserItemPrediction] = model.recommendForItem(featureRdds, 5)
      userRecs.take(10).foreach(println)
      itemRecs.take(10).foreach(println)

      sc.stop()
    } catch {
      case e: Exception => e.printStackTrace()
        System.exit(0)
    }
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String): (DataFrame, Int, Int, Int) = {
    import sqlContext.implicits._
    /**
      * Company(company_name: String, score: Int, status: String, company_type: String, industry: String, registered_capital: String,
      * registered_time: String, employe_scale: String, contributed_capital: String, vc_step: String, label: Int, id: Int)
      */
    val ratingDF: DataFrame = sqlContext.read.parquet(dataPath + "tagpredict/3000_data/cleanWithTextClassification")
      .rdd.map(x => {
      Company(x.getString(0), x.getInt(1), x.getString(2), x.getString(3), x.getString(4), x.getString(5), x.getString(6), x.getString(7)
        , x.getString(8), x.getString(9), x.getInt(10), x.getString(11), x.getString(12), x.getInt(13), x.getInt(14))
    }).toDF()

    val minMaxCompany: Row = ratingDF.agg(count("id").cast(IntegerType), countDistinct("industry").cast(IntegerType),
      countDistinct("vc_step").cast(IntegerType)).collect()(0)
    val (id: Int, industry: Int, step: Int) = (minMaxCompany.getInt(0), minMaxCompany.getInt(1), minMaxCompany.getInt(2))
    (ratingDF, id, industry, step)
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
    val statusUDF = udf(TagUtils.company_status)
    val typeUDF = udf(TagUtils.company_type)
    val registered_timeUDF = udf(TagUtils.registered_time)
    val employee_scaleUDF = udf(TagUtils.employee_scale)
    val double2intUDF = udf(TagUtils.double2int)
    val icp_audit_timeUDF = udf(TagUtils.icp_audit_timeUDF)
    val status_posUDF = udf(TagUtils.status_posUDF)
    val status_negUDF = udf(TagUtils.status_negUDF)

    val userDfUse = ratingDF
      .withColumn("score-money", bucketUDF(col("score"), col("registered_capital")))
      .withColumn("registered_capital", moneyUDF(col("registered_capital")))
      .withColumn("contributed_capital", moneyUDF(col("contributed_capital")))
      .withColumn("status", statusUDF(col("status")))
      .withColumn("company_type", typeUDF(col("company_type")))
      .withColumn("registered_time", registered_timeUDF(col("registered_time")))
      .withColumn("employe_scale", employee_scaleUDF(col("employe_scale")))
      .withColumn("icp_audit_time_format", icp_audit_timeUDF(col("icp_audit_time_format")))
      .withColumn("status_pos", status_posUDF(col("status_codes")))
      .withColumn("status_neg", status_negUDF(col("status_codes")))
    //transform string2index
    val df1: DataFrame = industry_category(userDfUse)
    val df2: DataFrame = vc_step_category(df1)
      .withColumn("industry_id", double2intUDF(col("industry_id")))
      .withColumn("step_id", double2intUDF(col("step_id")))

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(ratingDF)
      negativeDF.unionAll(ratingDF.withColumn("label", lit(2)))
    }
    else df2
    // userId, itemId as embedding features
    val joined = unioned
      .select(col("id"), col("label"), col("score"), col("registered_capital"), col("score-money"),
        col("status"), col("company_type"), col("registered_time"), col("employe_scale"),
        col("contributed_capital"), col("industry_id"), col("step_id"), col("status_pos"), col("status_neg"), col("icp_audit_time_format"), col("text_label"))
    val rddOfSample: RDD[UserItemFeature[Float]] = joined.rdd.map(r => {
      val iid = r.getAs[Int]("id")
      UserItemFeature(1, iid, Utils.row2Sample(r, columnInfo, modelType))
    })
    rddOfSample
  }
}
