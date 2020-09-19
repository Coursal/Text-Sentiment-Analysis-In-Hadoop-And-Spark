import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark._
import org.apache.spark.SparkContext._

import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors

import org.apache.spark.mllib.evaluation.MulticlassMetrics

/*
Execution Guide:
sbt package
spark-submit --master local ./target/scala-2.12/nb_featsel_2.12-0.1.jar
*/

object NB_FeatSel
{
	def main(args: Array[String]): Unit = 
	{
		// create a scala spark context for rdd management and a spark session for dataframe management
		val conf = new SparkConf().setAppName("Naive Bayes with Feature Selection")
		val sc = new SparkContext(conf)
		val spark = SparkSession.builder.appName("Naive Bayes with Feature Selection").master("local").getOrCreate()

        val start_time = System.nanoTime();

		// read the .csv file with the training data 
		val train_input = sc.textFile("hdfs://localhost:9000/user/crsl/input/tweets.csv")
						.map(line => line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)"))		// split each line to columns..
						.map(column => 
									{	// map the cleaned up tweet text as key and sentiment as value
										(
				                            column(1).toDouble // set the sentiment of the tweet as key
				                            ,
				                            column(3) // set the cleaned up tweet text as value, by cleaning up the text from...
												.replaceAll("(http|https)\\:\\/\\/[a-zA-Z0-9\\-\\.]+\\.[a-zA-Z]{2,3}(\\/\\S*)?", "") // links...
				                                .replaceAll("#|@|&.*?\\s", "")  // mentions, hashtags, special characters...
				                                .replaceAll("\\d+", "")         // numbers...
				                                .replaceAll("[^a-zA-Z ]", "")   // punctuation...
				                                .toLowerCase()                  // turn every character left to lowercase...
				                                .trim()                         // trim the spaces before & after the whole string...
				                                .replaceAll("\\s+", " ")        // and get rid of double spaces
										)
									}
							)


		// convert the RDD type sets of training data to Dataframes with named columns in order to apply the TFIDF measure on them
		import spark.implicits._
		val train_input_dataframe = train_input.toDF("label", "tweet")

		// apply TFIDF to the text of training data to have the proper form of (double, Vectors(double[])) used at the Naive Bayes classifier 
		val train_tokenizer = new Tokenizer().setInputCol("tweet").setOutputCol("words")		// split the text of each tweet to tokens
    	val train_words_data = train_tokenizer.transform(train_input_dataframe)

    	val train_hashingTF = new HashingTF().setInputCol("words")setOutputCol("rawFeatures")	// calculate TF
      	val train_featurized_data = train_hashingTF.transform(train_words_data)

	    val train_idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")			// calculate IDF
	    val train_idf_model = train_idf.fit(train_featurized_data)

	    val train_rescaled_data = train_idf_model.transform(train_featurized_data)				// calculate TFIDF
	    train_rescaled_data.select("label", "features").show()





	    // select the top 20% (based on the Pareto principle) of the train features with the highest TFIDF scores 
	    val feature_selector = new ChiSqSelector().setPercentile(0.2)
											      .setFeaturesCol("features")
											      .setLabelCol("label")
											      .setOutputCol("selectedFeatures")

		val selected_train_features = feature_selector.fit(train_rescaled_data).transform(train_rescaled_data)
    	selected_train_features.show()





	    // read the .csv file with the testing data 
		val test_input = sc.textFile("hdfs://localhost:9000/user/crsl/test_data/test_data.csv")
						.map(line => line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)"))		// split each line to columns..
						.map(column => 
									{	// map the cleaned up tweet text as key and sentiment as value
										(
				                            column(1).toDouble // set the sentiment of the tweet as key
				                            ,
				                            column(3) // set the cleaned up tweet text as value, by cleaning up the text from...
												.replaceAll("(http|https)\\:\\/\\/[a-zA-Z0-9\\-\\.]+\\.[a-zA-Z]{2,3}(\\/\\S*)?", "") // links...
				                                .replaceAll("#|@|&.*?\\s", "")  // mentions, hashtags, special characters...
				                                .replaceAll("\\d+", "")         // numbers...
				                                .replaceAll("[^a-zA-Z ]", "")   // punctuation...
				                                .toLowerCase()                  // turn every character left to lowercase...
				                                .trim()                         // trim the spaces before & after the whole string...
				                                .replaceAll("\\s+", " ")        // and get rid of double spaces
										)
									}
							)


		// convert the RDD type sets to Dataframes with named columns in order to apply the TFIDF measure on them
		val test_input_dataframe = test_input.toDF("label", "tweet")

		// apply TFIDF to the tweet text to have the proper form of (double, Vectors(double[])) that is used at the Naive Bayes classifier 
		val test_tokenizer = new Tokenizer().setInputCol("tweet").setOutputCol("words")			// split the text of each tweet to tokens
    	val test_words_data = test_tokenizer.transform(test_input_dataframe)

    	val test_hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")	// calculate TF
      	val test_featurized_data = test_hashingTF.transform(test_words_data)

	    val test_idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")			// calculate IDF
	    val test_idf_model = test_idf.fit(test_featurized_data)

	    val test_rescaled_data = test_idf_model.transform(test_featurized_data)					// calculate TFIDF
	    test_rescaled_data.select("label", "features").show()





	    // create the Naive Bayes model, train it with the train data and classify/predict the test data 
	    val model = new NaiveBayes().fit(selected_train_features)
	    val predictions = model.transform(test_rescaled_data)

	    println("EXECUTION DURATION: " + (System.nanoTime() - start_time) / 1000000000F + " seconds");

	    predictions.show()	// select example rows of predictions to display

	



	    // select each (prediction, true label) set and compute the test error, convert them to RDD, and use the MulticlassMetrics class
	    // to output the confussion matrix and some metrics
	    val prediction_and_labels = predictions.select("prediction", "label").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    
    	val metrics = new MulticlassMetrics(prediction_and_labels)
    	println("CONFUSSION MATRIX:")
    	println(metrics.confusionMatrix)
    	println("ACCURACY: " + metrics.accuracy)
    	println("F1 SCORE: " + metrics.weightedFMeasure)
    	println("PRECISION: " + metrics.weightedPrecision)
    	println("SENSITIVITY: " + metrics.weightedTruePositiveRate)
    	
	    spark.stop()
		sc.stop()
	}
}

