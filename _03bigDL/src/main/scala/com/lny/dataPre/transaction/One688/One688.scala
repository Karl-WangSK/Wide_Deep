package com.lny.dataPre.transaction.One688

import com.lny.dataPre.data_interface.DataPrepare
import com.lny.utils.UDFunctions.{cleanScoreFunc, _}
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql._
import org.apache.spark.storage.StorageLevel

object One688 extends DataPrepare {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("1688datapre")
      .getOrCreate()
    try {
      //entry
      run(spark, args(0))

      spark.stop()
    } catch {
      case e: Exception =>
        e.printStackTrace()
        System.exit(1)
    }

  }

  def Clean_Raw_data(spark: SparkSession, prefix: String): Unit = {
    //register udf functions
    val cleanScore: UserDefinedFunction = udf(cleanScoreFunc)
    val trimnull: UserDefinedFunction = udf(trimnullFunc)
    val trimstrnull: UserDefinedFunction = udf(trimstrnullFunc)
    val change: UserDefinedFunction = udf(changeFunc)
    //读取所有features   （该表的feature从宽表里选取出来的）
    var read1: DataFrame = spark.read.parquet(prefix + "bigdl/tagpredict/1688_data/raw")
    read1 = read1
      .withColumn("zero", lit(0))
      .withColumn("id", row_number().over(Window.partitionBy("zero").orderBy(desc("company_name"))))
      .withColumn("1688_medal_of_trading", trimnull(col("1688_medal_of_trading")).cast(IntegerType))
      .withColumn("score", cleanScore(trimnull(col("score"))))
      .withColumn("1688_chengxintong", trimnull(col("1688_chengxintong")).cast(IntegerType))
      .withColumn("label", change(col("1688_authentication_information")))
      .withColumn("1688_sumSales", trimnull(col("1688_sumSales")))
      .withColumn("company_status", trimstrnull(col("company_status")))
      .withColumn("company_type", trimstrnull(col("company_type")))
      .withColumn("industry", trimstrnull(col("industry")))
      .withColumn("registered_capital", trimstrnull(col("registered_capital")))
      .withColumn("registered_time", trimstrnull(col("registered_time")))
      .withColumn("employe_scale", trimstrnull(col("employe_scale")))
      .withColumn("contributed_capital", trimstrnull(col("contributed_capital")))
      .withColumn("1688_business_model", trimstrnull(col("1688_business_model")))
      .drop("1688_authentication_information", "zero")
    //sink
    read1.write
      .mode(SaveMode.Overwrite)
      .parquet(prefix + "bigdl/tagpredict/1688_data/clean")

  }

  def Extract_Raw_data(spark: SparkSession, prefix: String): Unit = {
    val tianyan: DataFrame = spark.read
      .parquet(prefix + "dataPlatform/web/dtbird/view/test/view_company_base_test/")
    //选取字段
    val raw: DataFrame = tianyan.select("aaa")
    raw.write
      .mode(SaveMode.Overwrite)
      .parquet(prefix + "bigdl/tagpredict/1688_data/raw")
  }

  // extract class 1、2 and class 3  from clean data ,then split them into five-five-open  for our data training afterwards
  def Extract_dataForTraining(spark: SparkSession, prefix: String): Unit = {
    //load clean data
    var clean_data: DataFrame = spark.read.parquet(prefix + "bigdl/tagpredict/1688_data/clean")

    clean_data.persist(StorageLevel.MEMORY_AND_DISK_2)

    val nonshili: Dataset[Row] = clean_data.where("label=='3'")
    val shili: Dataset[Row] = clean_data.where("label!='3'")

    nonshili.sample(0.317)
      .union(shili)
      .coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .parquet(prefix + "bigdl/tagpredict/1688_data/sampleForTrain")

  }

  override def SaveAsCsv(spark: SparkSession, prefix: String): Unit = ???
}
