import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql._
import org.apache.spark.rdd.RDD
import sqlContext.implicits._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.Row

val file = sc.textFile("hdfs://sandbox.hortonworks.com:8020/MLDataSet/datalag1.csv")
val datafile = file.filter(_.length > 0)

val dataRDD = datafile.map{line => 
            val x = line.split(',')
            LabeledPoint(x(0).toInt, Vectors.dense(x(1).toDouble, x(2).toDouble, x(3).toDouble, x(4).toDouble, x(5).toDouble, x(6).toDouble, x(7).toDouble, 
            x(8).toDouble, x(9).toDouble, x(10).toDouble, x(11).toDouble, x(12).toDouble, x(13).toDouble, x(14).toDouble, x(15).toDouble, x(16).toDouble, 
            x(17).toDouble, x(18).toDouble, x(19).toDouble, x(20).toDouble, x(21).toDouble, x(22).toDouble, x(23).toDouble, x(24).toDouble, x(25).toDouble, 
            x(26).toDouble, x(27).toDouble, x(28).toDouble, x(29).toDouble, x(30).toDouble, x(31).toDouble, x(32).toDouble, x(33).toDouble, x(34).toDouble, 
            x(35).toDouble, x(36).toDouble, x(37).toDouble, x(38).toDouble, x(39).toDouble, x(40).toDouble, x(41).toDouble))}.cache()
val data = dataRDD.toDF()
val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
val trainset = splits(0).cache()
val testset = splits(1)
val training = trainset.toDF("label","features")

val trainer = new MultilayerPerceptronClassifier()
    .setBlockSize(128)
    .setSeed(1234L)

val pipeline = new Pipeline()
    .setStages(Array(trainer))

// Initialize a layerGrid. hidden layers: 2, 3, 4; hidden neurons: 12, 16, 20, 24, 28, 32, 36
val layerGrid = Array(Array(41, 12, 12, 2), Array(41, 16, 16, 2), Array(41, 20, 20, 2), Array(41, 24, 24, 2), Array(41, 28, 28, 2), Array(41, 32, 32, 2),
                Array(41, 36, 36, 2), Array(41, 12, 12, 12, 2), Array(41, 16, 16, 16, 2), Array(41, 20, 20, 20, 2), Array(41, 24, 24, 24, 2), Array(41, 28, 28, 28, 2), 
                Array(41, 32, 32, 32, 2), Array(41, 36, 36, 36, 2), Array(41, 12, 12, 12, 12, 2), Array(41, 14, 14, 14, 14, 2), Array(41, 16, 16, 16, 16, 2), 
                Array(41, 20, 20, 20, 20, 2), Array(41, 24, 24, 24, 24, 2), Array(41, 28, 28, 28, 28, 2), Array(41, 32, 32, 32, 32, 2), Array(41, 36, 36, 36, 36, 2))

val paramGrid = new ParamGridBuilder()
    .addGrid(trainer.layers, layerGrid)
    .addGrid(trainer.maxIter, Array(100, 400, 800))
    .build()

val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(new MulticlassClassificationEvaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(3) 

val bestResult = cvModel.getEstimatorParamMaps
                .zip(cvModel.avgMetrics)
                .maxBy(_._2)
val acc = bestResult._2

val predictionAndLabel = cvModel.bestModel.transform(testset)
    .select("prediction", "label")
val accuracy = predictionAndLabel.withColumn("id", k("label")*0+1)
        .groupBy("prediction", "label").count()
accuracy.toDF.registerTempTable("acc")
println("Best layer settings: " + bestResult._1.apply(trainer.layers).mkString(" "))
println("Best Iterations: " + bestResult._1.apply(trainer.maxIter))
println("Best result: " + acc)
accuracy.show()

%sql
select * from acc




