import org.apache.commons.lang3.StringUtils
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object GenerateData {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .master("local[4]")
      .appName("test")
      .getOrCreate()
    run(spark, "id", "name")

  }


  def run(spark: SparkSession, str: String*): Unit = {
    import spark.implicits._
    val df: DataFrame = Seq(("1", "aaa"), ("2", "bbb"), ("3", "ccc")).toDF("id", "name")

    val arr: Array[String] = str.mkString(",").split(",")
    val col1 = arr(0)
    val col2 = arr(1)
    println(Seq(("1", "aaa"), ("2", "bbb"), ("3", "ccc"))(1))

    println(str)
  }

}
