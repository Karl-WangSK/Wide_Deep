package com.lny.dataPre.transaction.kehu3000

import com.lny.dataPre.data_interface.{Classification_Text, TCParams}

object Three3000TextClassification extends Classification_Text {

  def main(args: Array[String]): Unit = {
    //entry
    run(TCParams(
      need_classify = Seq("address","business_scope","introduce"),
      label = "label",
      transaction="3000",
      input_dir = "bigdl/tagpredict/3000_data/clean"
    ))

  }



}
