package com.lny.dataPre.transaction.kehu3000

import com.lny.dataPre.data_interface.DataPrepare
import com.lny.utils.UDFunctions.{changeFunc, cleanScoreFunc, trimnullFunc, trimstrnullFunc}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions._

object Three000 extends DataPrepare{
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("3000datapre")
      .getOrCreate()
    try {
      //entry
      run(spark,args(0))

      spark.sparkContext.stop()
    } catch {
      case e: Exception =>
        e.printStackTrace()
        System.exit(1)
    }
  }

  def Extract_dataForTraining(spark: SparkSession, prefix: String): Unit = {
    val clean: DataFrame = spark.read.parquet(prefix + "bigdl/tagpredict/3000_data/clean")
    clean.where("label=='2'")
      .limit(3000)
      .union(clean.where("label=='1'"))
      .write
      .mode(SaveMode.Overwrite)
      .parquet(prefix + "bigdl/tagpredict/3000_data/sampleForTrain")
  }


  def Clean_Raw_data(spark: SparkSession, prefix: String): Unit = {
    //register udf functions
    val cleanScore: UserDefinedFunction = udf(cleanScoreFunc)
    val trimnull: UserDefinedFunction = udf(trimnullFunc)
    val trimstrnull: UserDefinedFunction = udf(trimstrnullFunc)
    val change: UserDefinedFunction = udf(changeFunc)

    //加载icp日期
    val icp: DataFrame = spark.read.parquet(prefix+"dataPlatform/etl/dtbird/process/icp/icp_v1/")
      .select("company_name", "icp_audit_time_format")
    //加载网站信息
    val website: DataFrame = spark.read.parquet(prefix + "bigdl/tagpredict/3000_data/website_status")
    //load raw data
    val raw_data: DataFrame = spark.read.parquet(prefix + "bigdl/tagpredict/3000_data/raw")
      .select("company_name", "score", "company_status", "company_type", "industry", "registered_capital", "registered_time"
        , "employe_scale", "contributed_capital", "vc_step", "label")
      .join(icp, Seq("company_name"), "left_outer")
      .join(website, Seq("company_name"), "left_outer")

    val re: DataFrame = raw_data
      .withColumn("zero", lit(0))
      .withColumn("id", row_number().over(Window.partitionBy("zero").orderBy(desc("company_name"))))
      .withColumn("score", cleanScore(trimnull(col("score"))))
      .withColumn("company_status", trimstrnull(col("company_status")))
      .withColumn("company_type", trimstrnull(col("company_type")))
      .withColumn("industry", trimstrnull(col("industry")))
      .withColumn("registered_capital", trimstrnull(col("registered_capital")))
      .withColumn("registered_time", trimstrnull(col("registered_time")))
      .withColumn("employe_scale", trimstrnull(col("employe_scale")))
      .withColumn("contributed_capital", trimstrnull(col("contributed_capital")))
      .withColumn("icp_audit_time_format", trimstrnull(col("icp_audit_time_format")))
      .withColumn("vc_step", trimstrnull(col("vc_step")))
      .withColumn("status_codes", trimstrnull(col("status_codes")))
      .drop("zero")

    re
      .write
      .mode(SaveMode.Overwrite)
      .parquet(prefix + "bigdl/tagpredict/3000_data/clean")

  }


  def Extract_Raw_data(spark: SparkSession, prefix: String): Unit = {
    import spark.implicits._
    //加载客户给我们的数据
    val positive: DataFrame = spark.sparkContext.textFile(prefix + "bigdl/tagpredict/3000_data/kehu").toDF("company_name")
      .withColumn("label", lit(1))
    val kehu: Array[String] = spark.sparkContext.textFile(prefix + "bigdl/tagpredict/3000_data/kehu").collect()

    val tianyan: DataFrame = spark.read
      .parquet(prefix+"dataPlatform/web/dtbird/view/test/view_company_base_test/")
    //去掉正例的公司
    val negative: Dataset[Row] = tianyan
      .where("zhanzhangzhijia_location=='湖北' and  fiveone_url is null ")
      .filter { data =>
        var boo = true
        for (elem <- kehu) {
          if (data.getAs[String]("company_name") == elem) {
            boo = false
          }
        }
        boo
      }
      .select("company_name")
      .withColumn("label", lit(2))

    tianyan.join(positive.union(negative), Seq("company_name"), "inner")
      .dropDuplicates("company_name")
      .write
      .mode(SaveMode.Overwrite)
      .parquet(prefix + "bigdl/tagpredict/3000_data/raw")

  }

  def SaveAsCsv(spark: SparkSession, prefix: String): Unit = {
    import spark.implicits._
    val df1: RDD[String] = spark.sparkContext.textFile(prefix + "bigdl/tagpredict/3000_data/Predict_result/")
    val need_predict: DataFrame = spark.read.parquet(prefix + "bigdl/tagpredict/3000_data/cleanWithTextClassification/").select("company_name", "id", "label","status_codes").withColumnRenamed("id", "itemId")
    val tianyan: DataFrame = spark.read
      .parquet(prefix+"dataPlatform/etl/dtbird/clean/tianyan/tianyan_v1/")
      .drop("vc_course")
    //加载网站信息
    val predict: DataFrame = df1.map { data =>
      val strs: Array[String] = data.split(",")
      (strs(0).replace("UserItemPrediction(", ""), strs(1), strs(2), strs(3).replace(")", ""))
    }.map(data => (data._1, data._2, data._3, data._4.replace(")", "")))
      .toDF("userId", "itemId", "predict", "possibility")

    val temp: DataFrame = predict.join(need_predict, Seq("itemId"), "inner")

    tianyan.join(temp, Seq("company_name"), "right_outer")
      .coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", true)
      .csv(prefix + "bigdl/tagpredict/3000_data/resultCsv/")
  }

}
