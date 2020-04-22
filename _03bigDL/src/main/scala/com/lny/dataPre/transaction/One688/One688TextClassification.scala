package com.lny.dataPre.transaction.One688

import com.lny.dataPre.data_interface.{Classification_Text, TCParams}

object One688TextClassification extends Classification_Text {

  def main(args: Array[String]): Unit = {
    //entry
    run(TCParams(
      need_classify = Seq("introduce"),
      label = "1688_authentication_information",
      transaction="1688",
      input_dir = "bigdl/tagpredict/1688_data/clean"
    ))

  }


}
