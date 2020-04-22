package com.lny.dataPre.data_interface

import com.lny.utils.ForFastextData
import com.lny.utils.UDFunctions._
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  *
  * @param need_classify 需要做分类的列
  * @param label         标签
  * @param prefix        目录前缀
  * @param transaction   业务名
  * @param input_dir     输入数据的目录
  */
case class TCParams(
                     need_classify: Seq[String],
                     label: String,
                     prefix: String = "oss://LTAIQENhPMB5rMgl:AgCXsxreFgjXc12KhJtUtrRJNcV40T@dtbird-platform.oss-cn-shanghai-internal.aliyuncs.com/",
                     transaction: String,
                     input_dir: String
                   )

trait Classification_Text {
  //  替换null为空
  val trimstrnull: UserDefinedFunction = udf(trimstrnullFunc)

  def run(tCParams: TCParams): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("textpre")
      .getOrCreate()
    try {
      //load input data
      val input: DataFrame = spark.read.parquet(tCParams.prefix + tCParams.input_dir)
      //挑选出需要做文本分类的字段
      var classify_col: DataFrame = null
      if (tCParams.need_classify.length==1) {
        classify_col = spark.read
          .parquet(tCParams.prefix + "dataPlatform/web/dtbird/view/test/view_company_base_test/").select("company_name", s"${tCParams.need_classify.head}")
          .withColumn("text", trimstrnull(col(s"${tCParams.need_classify.head}")))
      } else if (tCParams.need_classify.length == 2) {
        classify_col = get_classify_col2(spark, tCParams)
      } else if (tCParams.need_classify.length == 3) {
        classify_col = get_classify_col3(spark, tCParams)
      }

      //对label和need_classify的列重命名 ，并加上ID
      var clean = input.join(classify_col, Seq("company_name"), "left_outer")
        .select("company_name", "text", s"${tCParams.label}")
        .withColumnRenamed(s"${tCParams.label}", "label")
        .withColumn("zero", lit(0))
        .withColumn("id", row_number().over(Window.partitionBy("zero").orderBy("company_name")).cast(StringType))
        .drop("zero")

      //tokenize
      ForFastextData.tokenizeForFasttext(spark, clean, tCParams.prefix, tCParams.transaction)

      spark.sparkContext.stop()
    } catch {
      case e: Exception =>
        e.printStackTrace()
        System.exit(1)
    }
  }

  def get_classify_col2(spark: SparkSession, tCParams: TCParams): DataFrame = {
    spark.read
      .parquet(tCParams.prefix + "dataPlatform/web/dtbird/view/test/view_company_base_test/").select("company_name", s"${tCParams.need_classify.head}",
      s"${tCParams.need_classify(1)}")
      .withColumn("text", concat_ws(",", trimstrnull(col(s"${tCParams.need_classify.head}")), trimstrnull(col(s"${tCParams.need_classify(1)}"))))
  }

  def get_classify_col3(spark: SparkSession, tCParams: TCParams): DataFrame = {
    spark.read
      .parquet(tCParams.prefix + "dataPlatform/web/dtbird/view/test/view_company_base_test/").select("company_name", s"${tCParams.need_classify.head}",
      s"${tCParams.need_classify(1)}", s"${tCParams.need_classify(2)}")
      .withColumn("text", concat_ws(",", trimstrnull(col(s"${tCParams.need_classify.head}")), trimstrnull(col(s"${tCParams.need_classify(1)}")),
        trimstrnull(col(s"${tCParams.need_classify(2)}"))))
  }


}
