package com.lny.dataPre.data_interface

import org.apache.spark.sql.SparkSession

trait DataPrepare {

  private val prefix = "oss://LTAIQENhPMB5rMgl:AgCXsxreFgjXc12KhJtUtrRJNcV40T@dtbird-platform.oss-cn-shanghai-internal.aliyuncs.com/"

  def run(spark:SparkSession,param:String): Unit ={
    param match {
      case "raw" =>
        //TODO：1、加载数据
        Extract_Raw_data(spark, prefix)
      case "clean" =>
        //TODO：2、数据清洗
        Clean_Raw_data(spark, prefix)
      case "sample" =>
        //TODO:3、取样训练
        Extract_dataForTraining(spark, prefix)
      case "evaluate" =>
        //TODO:4、将全集预测结果的公司找出 并存为csv
        SaveAsCsv(spark, prefix)
      case "all" =>
        //TODO：1、加载数据
        Extract_Raw_data(spark, prefix)
        //TODO：2、数据清洗
        Clean_Raw_data(spark, prefix)
        //TODO:3、取样训练
        Extract_dataForTraining(spark, prefix)
        //TODO:4、将全集预测结果的公司找出 并存为csv
        SaveAsCsv(spark, prefix)
      case _ => throw new IllegalArgumentException
    }
  }

  /**
    * 提取需要的feature
    * @param spark
    * @param prefix
    */
  def Extract_Raw_data(spark: SparkSession, prefix: String=prefix)

  /**
    * 预处理feature
    * @param spark
    * @param prefix
    */
  def Clean_Raw_data(spark: SparkSession, prefix: String=prefix)

  /**
    * 提取需要训练的数据
    * @param spark
    * @param prefix
    */
  def Extract_dataForTraining(spark: SparkSession, prefix: String=prefix)

  /**
    * 将结果存为CSV
    * @param spark
    * @param prefix
    */
  def SaveAsCsv(spark: SparkSession, prefix: String=prefix)

}
