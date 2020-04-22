package com.lny.utils

import java.util

import com.huaban.analysis.jieba.JiebaSegmenter
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.collection.JavaConverters._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

object ForFastextData {
  private val prefix = "oss://LTAIQENhPMB5rMgl:AgCXsxreFgjXc12KhJtUtrRJNcV40T@dtbird-platform.oss-cn-shanghai-internal.aliyuncs.com/"

  /**
    * 提取文本数据 供fasttext训练
    *
    * @param spark  sparksession
    * @param df   input DF
    * @param prefix  endpoint
    * @param transaction 业务名
    */
  def tokenizeForFasttext(spark: SparkSession, df: DataFrame, prefix: String = prefix, transaction: String): Unit = {
    import spark.implicits._
    //1、tokenize words
    // 2、assemble label and tokenized words
    val re: RDD[(String, String, String)] = df
      .rdd
      .map { data =>
        val introduce: String = data.getAs[String]("text")
        val label: Int = data.getAs[Int]("label")
        val name: String = data.getAs[String]("company_name")
        val id: String = data.getAs[String]("id")
        val jieba = new JiebaSegmenter()
        val strs: util.List[String] = jieba.sentenceProcess(introduce)
        val v: String = strs.asScala.mkString(" ").replace("\n", "").replace("\r\n", "").replace("\r", "")
        if (v == "") {
          (id, name, "")
        } else {
          (id, name, "__label__" + label.toString + " " + v)
        }
      }.coalesce(1)
    //cache
    re.persist(StorageLevel.MEMORY_AND_DISK_2)
    //all of id and company_name
    val id_n_name_dir = s"bigdl/tagpredict/textClassification/$transaction/id_n_companyName/"
    OSSUtils.deleteOssDir(id_n_name_dir)
    re.map(data => (data._1, data._2))
      .saveAsTextFile(prefix + id_n_name_dir)
    //all of  introduce
    val introduce_all = s"bigdl/tagpredict/textClassification/$transaction/introduce_all/"
    OSSUtils.deleteOssDir(introduce_all)
    re.map(_._3)
      .saveAsTextFile(prefix + introduce_all)
    //filter  positive
    val positive = re.filter(_._3 != "")
      .map(_._3)
      .filter(data => data.contains("__label__1")).toDF("v")
    //filter negative
    val negative = re.filter(_._3 != "")
      .map(_._3)
      .filter(data => data.contains("__label__2")).toDF("v")
    //union
    //for fasttext train  data
    negative.limit(positive.count().toInt)
      .union(positive)
      .sample(0.75)
      .coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .text(prefix + s"bigdl/tagpredict/textClassification/$transaction/introduce_train")
    //for fasttext valid data
    negative.limit(positive.count().toInt)
      .union(positive)
      .sample(0.25)
      .coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .text(prefix + s"bigdl/tagpredict/textClassification/$transaction/introduce_valide")
  }


  /**
    * 从fasttext训练完后提取文本预测标签
    *
    * @param spark
    * @param prefix
    * @param transaction
    */
  def extract_label(spark: SparkSession, prefix: String = prefix, transaction: String) = {
    val predict_label: RDD[String] = spark.sparkContext.textFile(prefix + s"bigdl/tagpredict/textClassification/$transaction/result")
    import spark.implicits._
    predict_label.map { data =>
      val arr: Array[String] = data.split("\\s+")
      (arr(0).split(",")(1).replace(")", ""), arr(1).split("__")(2))
    }
      .toDF("company_name", "text_label")
      .write
      .mode(SaveMode.Overwrite)
      .parquet(prefix + s"bigdl/tagpredict/${transaction}_data/introduce_label")

  }

  /**
    * 把预测的文本标签加到需要做训练的表中
    *
    * @param spark
    * @param prefix
    * @param transaction
    */
  def attach_label(spark: SparkSession, prefix: String = prefix, transaction: String) = {
    val intro_label: DataFrame = spark.read.parquet(prefix + s"bigdl/tagpredict/${transaction}_data/introduce_label")
    val trans_train: DataFrame = spark.read.parquet(prefix + s"bigdl/tagpredict/${transaction}_data/sampleForTrain/")
    val trans_clean: DataFrame = spark.read.parquet(prefix + s"bigdl/tagpredict/${transaction}_data/clean/")

    trans_train.join(intro_label, Seq("company_name"), "inner")
      .withColumn("text_label", col("text_label").cast(IntegerType))
      .write
      .mode(SaveMode.Overwrite)
      .parquet(prefix + s"bigdl/tagpredict/${transaction}_data/sampleWithTextClassification/")

    trans_clean.join(intro_label, Seq("company_name"), "inner")
      .withColumn("text_label", col("text_label").cast(IntegerType))
      .write
      .mode(SaveMode.Overwrite)
      .parquet(prefix + s"bigdl/tagpredict/${transaction}_data/cleanWithTextClassification/")
  }
}
