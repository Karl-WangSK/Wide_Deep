package com.lny.utils

import org.apache.commons.lang3.StringUtils
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.sql.DataFrame

import scala.util.matching.Regex

object TagUtils {

  // transform capital money to  number,which from 1 to 10 level
  def moneyTransform(): String => Int = {
    val func = (moneystr: String) =>
      if (StringUtils.isNotBlank(moneystr)) {
        val regex = new Regex("\\d+\\.?\\d+")
        val find: Option[String] = regex.findFirstIn(moneystr)
        if (find.isDefined) {
          val money: Int = find.get.toDouble.toInt
          if (money <= 1) 1
          else if (money > 1 && money <= 5) 2
          else if (money > 1 && money <= 5) 3
          else if (money > 5 && money <= 20) 4
          else if (money > 20 && money <= 50) 5
          else if (money > 50 && money <= 100) 6
          else if (money > 100 && money <= 500) 7
          else if (money > 500 && money <= 2000) 8
          else if (money > 2000 && money <= 10000) 9
          else if (money > 10000) 10
          else 0
        } else {
          0
        }
      } else {
        0
      }

    func
  }

  def chengxintong(): Int => Int = {
    val func = (chengxin: Int) =>
      if (chengxin == 1) 1
      else if (chengxin == 2) 2
      else if (chengxin == 3) 3
      else if (chengxin == 4) 4
      else if (chengxin == 5) 5
      else if (chengxin == 6) 6
      else if (chengxin == 7) 7
      else if (chengxin == 8) 8
      else if (chengxin == 9) 9
      else if (chengxin > 9 && chengxin <= 13) 10
      else if (chengxin > 13 && chengxin <= 16) 11
      else if (chengxin > 16) 12
      else 0
    func
  }

  def sumsales = (sales: Int) => {
    if (sales < 1000) 1
    else if (sales >= 1000 && sales < 5000) 2
    else if (sales >= 5000 && sales < 10000) 3
    else if (sales >= 10000 && sales < 50000) 4
    else if (sales >= 50000 && sales < 100000) 5
    else if (sales >= 100000 && sales < 1000000) 6
    else if (sales >= 1000000 && sales < 5000000) 7
    else if (sales >= 5000000 && sales < 10000000) 8
    else if (sales >= 10000000 && sales < 100000000) 9
    else if (sales >= 100000000 && sales < 1000000000) 10
    else 11
  }

  def medal(): Int => Int = {
    val func = (medal: Int) =>
      if (medal == 1) 1
      else if (medal == 2) 2
      else if (medal == 3) 3
      else if (medal == 4) 4
      else if (medal == 5) 5
      else 0
    func
  }

  def supply_level(): String => Int = {
    val func = (level: String) =>
      if (StringUtils.isNotBlank(level)) {
        if (level == "10万-20万") 1
        else if (level == "20万-50万") 2
        else if (level == "50万-120万") 3
        else if (level == "120万-300万") 4
        else if (level == "300万-800万") 5
        else if (level == "0-500") 6
        else if (level == "500-5000") 7
        else if (level == "5000-2万") 8
        else if (level == "2万-5万") 9
        else if (level == "5万-10万") 10
        else if (level == "800万-2000万") 11
        else if (level == "2000万-5000万") 12
        else if (level == "5000万-1.5亿") 13
        else if (level == "1.5亿-5亿") 14
        else if (level == "5亿以上") 15
        else 0
      } else {
        0
      }
    func
  }

  def company_status = (text: String) => {
    if (StringUtils.isNotBlank(text)) {
      if (text == "在业") 1
      else if (text == "小微企业") 2
      else if (text == "迁出") 3
      else if (text == "开业") 4
      else if (text == "注销") 5
      else if (text == "存续") 6
      else if (text == "解散") 7
      else if (text == "已告解散") 8
      else if (text == "仍注册") 9
      else 0
    } else {
      0
    }
  }


  def company_type = (text: String) => {
    if (StringUtils.isNotBlank(text)) {
      if (text == "联营企业") 1
      else if (text == "普通合伙") 2
      else if (text == "国有企业") 3
      else if (text == "有限责任公司") 4
      else if (text == "外商投资企业") 5
      else if (text == "有限合伙") 6
      else if (text == "股份有限公司") 7
      else if (text == "个人独资企业") 8
      else if (text == "个体工商户") 9
      else if (text == "私营企业") 10
      else if (text == "集体所有制") 11
      else 0
    } else {
      0
    }

  }

  def employee_scale = (text: String) => {
    if (StringUtils.isNotBlank(text)) {
      if (text == "0-49人") 1
      else if (text == "50-99人") 2
      else if (text == "500-999人") 3
      else if (text == "100-499人") 4
      else if (text == "1000-4999人") 5
      else if (text == "5000-9999人") 6
      else 0
    } else {
      0
    }

  }

  def registered_time = (text: String) => {
    if (StringUtils.isNotBlank(text)) {
      val regex = new Regex("\\d{4}-\\d{2}-\\d{2}")
      val find: Option[String] = regex.findFirstIn(text)
      if (find.isDefined) {
        val str: String = find.get
        val time = str.replace("-", "").toInt
        if (time < 2000) 1
        else if (time >= 2000 && time < 2005) 2
        else if (time >= 2005 && time < 2010) 3
        else if (time >= 2010 && time < 2012) 4
        else if (time >= 2012 && time < 2014) 5
        else if (time >= 2014 && time < 2016) 6
        else if (time >= 2016 && time < 2018) 7
        else if (time >= 2018 && time < 2020) 8
        else 9
      } else {
        0
      }
    } else {
      0
    }
  }



  def icp_audit_timeUDF = (text: String) => {
    if (StringUtils.isNotBlank(text)) {
      val regex = new Regex("\\d{4}-\\d{2}-\\d{2}")
      val find: Option[String] = regex.findFirstIn(text)
      if (find.isDefined) {
        val str: String = find.get
        val time = str.replace("-", "").toInt
        if (time < 2015) 1
        else if (time >= 2015 && time < 2016) 2
        else if (time >= 2016 && time < 2017) 3
        else if (time >= 2017 && time < 2018) 4
        else if (time >= 201806 && time < 2019) 5
        else if (time >= 201906 && time < 2020) 6
        else 7
      } else {
        0
      }
    } else {
      0
    }
  }

  //对website_可访问的网站数分类  word2index
  def status_posUDF = (text: String) => {
    if (StringUtils.isNotBlank(text)) {
      val arr: Array[String] = text.split("-")
      val number: Int = arr(1).toInt
      if (number == 0) 1
      else if (number == 1) 2
      else if (number == 2) 3
      else if (number == 3) 4
      else if (number == 4) 5
      else if (number >= 5) 6
      else 7
    } else {
      0
    }
  }

  //对website不可访问的网站数分类  word2index
  def status_negUDF = (text: String) => {
    if (StringUtils.isNotBlank(text)) {
      val arr: Array[String] = text.split("-")
      val number: Int = arr(0).toInt - arr(1).toInt
      if (number == 0) 1
      else if (number == 1) 2
      else if (number == 2) 3
      else if (number == 3) 4
      else if (number == 4) 5
      else if (number >= 5) 6
      else 7
    } else {
      0
    }
  }

  def bussiness_model = (text: String) => {
    if (StringUtils.isNotBlank(text)) {
      if (text == "生产厂家") 1
      else if (text == "经销批发") 2
      else if (text == "招商代理") 3
      else if (text == "商业服务") 4
      else 0
    } else {
      0
    }
  }

  def double2int = (text: Double) => {
    text.toInt
  }


  //对industry分类  word2index
  def industry_category(df: DataFrame): DataFrame = {
    val indexer: StringIndexer = new StringIndexer()
      .setInputCol("industry")
      .setOutputCol("industry_id")
    val model: StringIndexerModel = indexer.fit(df)
    model.transform(df)
  }

  //对vc_step分类  word2index
  def vc_step_category(df: DataFrame): DataFrame = {
    val indexer: StringIndexer = new StringIndexer()
      .setInputCol("vc_step")
      .setOutputCol("step_id")
    val model: StringIndexerModel = indexer.fit(df)
    model.transform(df)
  }


  //对city分类  word2index
  def city_category(df: DataFrame): DataFrame = {
    val indexer: StringIndexer = new StringIndexer()
      .setInputCol("city")
      .setOutputCol("city_id")
    val model: StringIndexerModel = indexer.fit(df)
    model.transform(df)
  }

  //对province分类  word2index
  def province_category(df: DataFrame): DataFrame = {
    val indexer: StringIndexer = new StringIndexer()
      .setInputCol("province")
      .setOutputCol("province_id")
    val model: StringIndexerModel = indexer.fit(df)
    model.transform(df)
  }


}
