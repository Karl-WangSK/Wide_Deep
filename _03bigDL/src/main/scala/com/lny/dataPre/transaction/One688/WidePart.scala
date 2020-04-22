package com.lny.dataPre.transaction.One688

import java.io.InputStream
import java.util
import java.util.Scanner

import com.huaban.analysis.jieba.JiebaSegmenter
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, SaveMode, SparkSession}

import scala.collection.mutable

object WidePart {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("widepart")
      .getOrCreate()
    import spark.implicits._
    try {
      //从词袋中获取该词对应的index
      val getWideCol = udf((text: String, index: Int) => {
        text.split(",")(index)
      })

      var readDF: DataFrame = spark.read
        .parquet("oss://LTAIQENhPMB5rMgl:AgCXsxreFgjXc12KhJtUtrRJNcV40T@dtbird-platform.oss-cn-shanghai-internal.aliyuncs.com/dataPlatform/web/dtbird/view/test/view_company_base_test/")
        .where("1688_url is not null")
      //假设已筛选出需要深度学习的col
      val fieldsName: Array[String] = readDF
        .select("company_name", "introduce", "address", "company_type", "industry").schema.fieldNames
      //合并所有col
      val contactsCol: Array[Column] = fieldsName.filter(_ != "company_name").map(str => col(str))
      var contactcol: Column = lit("")
      for (contact <- contactsCol) {
        contactcol = concat_ws(",", contact, contactcol)
      }
      //load stopwords
      val set = new mutable.HashSet[String]()
      val in: InputStream = WidePart.getClass.getClassLoader.getResourceAsStream("stop_words.txt")
      val scanner = new Scanner(in)
      while (scanner.hasNext()){
        set.add(scanner.nextLine())
      }

      val stop_words: Broadcast[mutable.HashSet[String]] = spark.sparkContext.broadcast(set)
      //提取合并了的字段
      import scala.collection.JavaConverters._
      var wordsDS: RDD[(String, String)] = readDF
        .withColumn("allCols", contactcol)
        .select("company_name", "allCols")
        .rdd
        .mapPartitions(_.map { data =>
          val cols: String = data.getAs[String]("allCols")
          val company_name: String = data.getAs[String]("company_name")
          //结巴分词
          val list: util.List[String] = new JiebaSegmenter().sentenceProcess(cols)
          val re: mutable.Buffer[String] = list.asScala.filterNot(stop_words.value)
          (company_name, re.mkString(","))
        })


      //词袋
      val wordsArr: Array[String] = wordsDS.map(_._2).collect().mkString(",").split(",").toSet[String].toArray[String]
      //获取30列作为wideCol
      var re = wordsDS.mapPartitions(_.map { data =>
        val cols: Array[String] = data._2.split(",")
        var num = ""
        for (i <- 0 until 40) {
          val str: String = FromVocabList(cols, i, wordsArr)
          num += "," + str
        }
        if (num.startsWith(",")) {
          (data._1, num.substring(1))
        } else {
          (data._1, num)
        }
      }).toDF("company_name", "allCols")
      //转为多列
      for (i <- 0 until 40) {
        re = re.withColumn(s"wide${i}", getWideCol(col("allCols"), lit(i)))
      }

      //sink
      re
        .write
        .mode(SaveMode.Overwrite)
        .parquet("oss://LTAIQENhPMB5rMgl:AgCXsxreFgjXc12KhJtUtrRJNcV40T@dtbird-platform.oss-cn-shanghai-internal.aliyuncs.com/bigdl/tagpredict/1688_data/1688_wideCols")
      //stop
      spark.sparkContext.stop()
    } catch {
      case e: Exception => e.printStackTrace()
        System.exit(1)
    }
  }

  def FromVocabList(text: Array[String], index: Int, vocabList: Array[String]) = {
    val default: Int = 0
    val start: Int = 1
    if (text.length >= index + 1 && vocabList.contains(text(index))) (vocabList.indexOf(text(index)) + start).toString
    else default.toString
  }

}
