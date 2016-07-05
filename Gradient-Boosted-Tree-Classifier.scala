import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

val file = sc.textFile("hdfs://sandbox.hortonworks.com:8020/MLDataSet/datalag1.csv")
val datafile = file.filter(_.length > 0)

val data = datafile.map{line => val x = line.split(',')
            LabeledPoint(x(0).toInt, Vectors.dense(x(1).toDouble, x(2).toDouble, 
            x(3).toDouble, x(4).toDouble, x(5).toDouble, x(6).toDouble, x(7).toDouble, 
            x(8).toDouble, x(9).toDouble, x(10).toDouble, x(11).toDouble, x(12).toDouble, 
            x(13).toDouble, x(14).toDouble, x(15).toDouble, x(16).toDouble, x(17).toDouble, 
            x(18).toDouble, x(19).toDouble, x(20).toDouble, x(21).toDouble, x(22).toDouble, 
            x(23).toDouble, x(24).toDouble, x(25).toDouble, x(26).toDouble, x(27).toDouble, 
            x(28).toDouble, x(29).toDouble, x(30).toDouble, x(31).toDouble, x(32).toDouble, 
            x(33).toDouble, x(34).toDouble, x(35).toDouble, x(36).toDouble, x(37).toDouble, 
            x(38).toDouble, x(39).toDouble, x(40).toDouble, x(41).toDouble))}.cache()
            
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)

val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

val gbt = new GBTClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setMaxIter(10)

val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

val model = pipeline.fit(trainingData)

val predictions = model.transform(testData)

predictions.select("predictedLabel", "label").groupBy("predictedLabel", "label").count().show()
val accuracy = predictions.withColumn("id", predictions("label")*0+1).groupBy("label", "predictedLabel").count().toDF.registerTempTable("prediction")

%sql
select * from prediction limit 1000

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("precision")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println("Learned classification GBT model:\n" + gbtModel.toDebugString)
