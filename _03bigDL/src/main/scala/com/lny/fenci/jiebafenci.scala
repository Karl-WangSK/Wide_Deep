package com.lny.fenci

import java.io.{File, FileWriter}

import com.huaban.analysis.jieba.JiebaSegmenter
import com.huaban.analysis.jieba.JiebaSegmenter.SegMode
import org.apache.spark.sql.functions._

import scala.collection.JavaConversions._
import scala.io.Source


object jiebafenci {
  def jieba(s: String): String = {
    val segmenter = new JiebaSegmenter()
    val str: String = segmenter.process(s, SegMode.SEARCH).toList.map(_.word + " ").foldLeft("")(_ + _)
    println(str)
    str
  }

  def jieba_udf = udf((s: String) => jieba(s))

  def stop_words(): Set[String] = {
    val in = getClass().getResourceAsStream("/stop_test.txt")
    Source.fromInputStream(in).getLines.toSet
  }

  def unit_test(): Unit = {
    //    val test =  getClass().getClassLoader().getResourceAsStream("userdic")
    val file = new File("D:\\2.txt")
    val out = new FileWriter(file)
    val sentences: Array[String] = Array[String]("这是一个伸手不见五指的黑夜。我叫孙悟空，我爱北京，我爱Python和C++。",
      "我不喜欢日本和服。", "雷猴回归人间。",
      "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
      "结果婚的和尚未结过婚的",
      "这很亦可赛艇")
    for (sentence <- sentences) {
      jieba(sentence).split(" ").filterNot(stop_words()).foreach(data =>
        println(data)
      )
    }
    out.close()
  }

  def main(args: Array[String]): Unit = {
    unit_test()

  }


}
