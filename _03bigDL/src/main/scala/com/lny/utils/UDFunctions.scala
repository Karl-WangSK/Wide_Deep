package com.lny.utils

import org.apache.commons.lang3.StringUtils
import org.apache.spark.sql.functions.udf

object UDFunctions {


  val cleanScoreFunc = (text: String) => {
    if (text == null) {
      0
    } else {
      text.replace("\\&quot;", "").toInt
    }
  }
  val trimnullFunc = (text: String) => {
    if (StringUtils.isNotBlank(text) && text != null) {
      text
    } else {
      "0"
    }
  }

  val trimstrnullFunc=(text:String)=>{
    if(StringUtils.isNotBlank(text)){
      text
    }else{
      ""
    }
  }

  val changeFunc=(text:Int)=>{
    if(text==0){
      3
    }else{
      text
    }
  }


}
