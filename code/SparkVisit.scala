import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
 
// Definition of Class used to parse the csv
case class Visit(id: String, feat1: Double, feat2: Double, feat3: Double, sane: Double)

/** Computes an approximation to pi */
object SparkVisit {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Spark Visit")
    val spark = new SparkContext(conf)
    // Parameters used:
    // max_glu_serum,A1Cresult,insulin, diabetesMed
    
    val basePath = if (args.length > 0) args(0).toString else ""

    // Load the file to a DataFrame
    val data = sc.textFile(basePath + "\\diab.csv")
    .filter(!_.contains("encounter_id"))
    .map(_.split(","))
    .map(p => Visit(p(1),convert(p(22)),convert(p(23)), convert(p(41)),checkSano(p(48))))
    .toDF("ID","Feat1", "Feat2", "Feat3", "Sano").cache()

    // Grouping by the number of visit of single user
    val raggr= data.groupBy($"ID").agg(count($"ID").alias("NumVisite"));

    // Creo i LabeledPoint con tutte le features
    val labeledAll = data.map(row => LabeledPoint(row.getDouble(4), Vectors.dense(row.getDouble(1), row.getDouble(2), row.getDouble(3))))

    // Labeled point feature 1 + 0
    val labeled_1 = data.map(row => LabeledPoint(row.getDouble(4), Vectors.dense(row.getDouble(1), 0.0)))

    // Labeled point feature 2 + 0
    val labeled_2 = data.map(row => LabeledPoint(row.getDouble(4), Vectors.dense(row.getDouble(2), 0.0)))

    // Labeled point feature 3 + 0
    val labeled_3 = data.map(row => LabeledPoint(row.getDouble(4), Vectors.dense(row.getDouble(3), 0.0)))

    // Labeled point feature 3 + 2

    val labeled_2_and_3 = data.map(row => LabeledPoint(row.getDouble(4), Vectors.dense(row.getDouble(3), row.getDouble(2))))
    //val model = SVMWithSGD.train(labeled, 100)

    //MLUtils.saveAsLibSVMFile(labeled, "C:\\Users\\geso8_000\\Documents\\esempio-spark\\LastSVML")

    val svm= new SVMWithSGD()
    svm.setIntercept(true)
    val model = svm.run(labeled_2_and_3)

    val scoreAndLabels = labeled.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    model.clearThreshold

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val areaUnderROC = metrics.areaUnderROC()
    println("Area under ROC = " + auROC)

    // ROC Curve
    val roc = metrics.roc.collect()
    
    // Save RDD in csv
    roc.map(line => line._1 +","+line._2).coalesce(1,true).saveAsTextFile(basePath + "\\roc")

    spark.stop()
  }

    // Function defined to return if the patient is sane or not
  def checkSano(s:String) : Double = {
    val myString = s.trim.toLowerCase;
    if(myString.equals("no")){
      return 0;
    } 
      return 1;
  }

  // Convert the value in the string to a Double
  def convert(s:String) : Double = {
    val myString = s.trim.toLowerCase;
    myString match {
      case "none" | "no" => 0
      case "normal" | "steady" => 1
      case ">200" | ">7" | "up" => 2
      case ">300" | ">8" | "down" => 3
      case _ => 0
    }
  }


}