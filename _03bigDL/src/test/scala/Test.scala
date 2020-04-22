import java.io.InputStream
import java.util.Scanner

import com.lny.dataPre.transaction.One688.WidePart
import org.apache.spark.mllib
import scala.collection.mutable
import scala.io.Source

object Test {

  def main(args: Array[String]): Unit = {
    val set = new mutable.HashSet[String]()
    val in: InputStream = WidePart.getClass.getClassLoader.getResourceAsStream("stop_words.txt")
    val scanner = new Scanner(in)
    while (scanner.hasNext()){
      set.add(scanner.nextLine())
    }
   val strs = List("尼玛炸l+,",",")
    println(strs.filterNot(set).mkString(","))


  }

}
