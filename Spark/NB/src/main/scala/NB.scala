import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark._
import org.apache.spark.SparkContext._

import org.apache.spark.rdd.RDD

import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.mllib.evaluation.MulticlassMetrics

/*
Execution Guide:
sbt package
spark-submit --master local ./target/scala-2.12/nb_2.12-0.1.jar spark_input_#
*/

object NB
{
    // function that splits each line of the .csv file by commas, while being cautious at the
    // commas that might exist inside the last quoted column of the line with the tweet text
    def split_csv(line:String) : Array[String] = 
    {
        var columns = line.split(",");

        // if the columns are more than 4, that means the text of the post had commas inside,  
        // so stitch the last columns together to form the full text of the tweet
        if(columns.length > 4)
        {
            for(i <- 4 until columns.length)
                columns(3) += columns(i);
        }

        return columns
    }

    def main(args: Array[String]): Unit = 
    {
        // create a scala spark context for rdd management and a spark session for dataframe management
        val conf = new SparkConf().setAppName("Naive Bayes")
        val sc = new SparkContext(conf)
        val spark = SparkSession.builder.appName("Naive Bayes").master("local").getOrCreate()

        val start_time = System.nanoTime()

        // read the .csv file with the training data 
        val input = sc.textFile("hdfs://localhost:9000/user/crsl/spark_input_" + args(0) + "/tweets.csv")
                        .map(line => split_csv(line))     // split each line to columns...
                        .map(column => 
                                    {   // map the cleaned up tweet text as key and sentiment as value
                                        (
                                            column(1).toDouble // set the sentiment of the tweet as key
                                            ,
                                            column(3) // set the cleaned up tweet text as value, by cleaning up the text from...
                                                .replaceAll("(?i)(https?:\\/\\/(?:www\\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\\.[^\\s]{2,}|www\\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\\.[^\\s]{2,}|https?:\\/\\/(?:www\\.|(?!www))[a-zA-Z0-9]+\\.[^\\s]{2,}|www\\.[a-zA-Z0-9]+\\.[^\\s]{2,})", "")
                                                .replaceAll("(#|@|&).*?\\w+", "")   // mentions, hashtags, special characters...
                                                .replaceAll("\\d+", "")             // numbers...
                                                .replaceAll("[^a-zA-Z ]", " ")      // punctuation...
                                                .toLowerCase()                      // turn every character left to lowercase...
                                                .trim()                             // trim the spaces before & after the whole string...
                                                .replaceAll("\\s+", " ")            // and get rid of double spaces
                                        )
                                    }
                            )


        // convert the RDD type sets of training data to Dataframes with named columns in order to apply the TFIDF measure on them
        import spark.implicits._
        val input_dataframe = input.toDF("label", "tweet")


        // apply TFIDF to the text of training data to have the proper form of (double, Vectors(double[])) used at the Naive Bayes classifier 
        val tokenizer = new Tokenizer().setInputCol("tweet").setOutputCol("words")      // split the text of each tweet to tokens
        val words_data = tokenizer.transform(input_dataframe)

        val input_hashingTF = new HashingTF().setInputCol("words")setOutputCol("rawFeatures")   // calculate TF
        val input_featurized_data = input_hashingTF.transform(words_data)

        val input_idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")           // calculate IDF
        val input_idf_model = input_idf.fit(input_featurized_data)

        val input_rescaled_data = input_idf_model.transform(input_featurized_data)              // calculate TFIDF
        //input_rescaled_data.select("label", "features").show()

        val Array(training_data, test_data) = input_rescaled_data.randomSplit(Array(0.75, 0.25), seed = 1234L)

        // create the Naive Bayes model, train it with the train data and classify/predict the test data 
        val model = new NaiveBayes().fit(training_data)
        val predictions = model.transform(test_data)
        //predictions.show()
        val end_time = System.nanoTime()


        // select each (prediction, true label) set and compute the test error, convert them to RDD, and use the MulticlassMetrics class
        // to output the confussion matrix and some metrics
        val prediction_and_labels = predictions.select("prediction", "label").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    
        val metrics = new MulticlassMetrics(prediction_and_labels)
        println(metrics.confusionMatrix)
        println("ACCURACY: " + metrics.accuracy)
        println("F1 SCORE: " + metrics.weightedFMeasure)
        println("PRECISION: " + metrics.weightedPrecision)
        println("SENSITIVITY: " + metrics.weightedTruePositiveRate)
        println("EXECUTION DURATION: " + (end_time - start_time) / 1000000000F + " seconds");

        spark.stop()
        sc.stop()   
    }
}