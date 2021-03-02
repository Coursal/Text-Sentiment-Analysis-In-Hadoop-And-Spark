# Text Sentiment Analysis In Hadoop & Spark

The source code developed and used for the purposes of my thesis with the same title under the guidance of my supervisor professor [Vasilis Mamalis](http://users.teiath.gr/vmamalis/) for the [Department of Informatics and Computer Engineering](http://www.ice.uniwa.gr/en/home/) of the [University of West Attica](https://www.uniwa.gr/en/).

You can read the text and presentation of the thesis and cite it as seen [here]() (with DOI provided, of course).

## Main Objectives
* studying the basics of text mining, sentiment analysis, MapReduce, and paraller/distributed computing
* developing a number of applications for sentiment analysis using the Apache Hadoop and Apache Spark frameworks in a cluster of computers using Hadoop as infrastructure
* analyzing the results both in classification (accuracy, F1 measure) from the confusion matrix) and parallel execution (execution time, scalability, speedup)
* coming to conclusions for each application's performance and proposing possible extensions to be made in the future

## Developed Applications
All the classifications models of the applications below use **75%** of the input for **training** and the rest **25%** for **testing** purposes.
#### Using the Hadoop framework
  * **Simple Version of Naive Bayes**
  * **Modified Version of Naive Bayes** (using TFIDF to implement feature selection by only using the top 75% most relevant features)
#### Using the Spark framework
  * **Simple Version of Naive Bayes**
  * **Modified Version of Naive Bayes** (using TFIDF to implement feature selection by only using features that are shown in 5 or more documents)
  * **Simple Version of SVM**
  * **Modified Version of SVM** (using TFIDF to implement feature selection by only using features that are shown in 5 or more documents)
  
 ## Input Data Used
 To train and test the classification models of each application, [this](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/) dataset of 1.6 million tweets was used in `csv` format with each line representing a document/tweet with its unique ID, class label, and tweet text as seen below.
 
 *image here*
 
For the Hadoop-based applications, 10 datasets that differ in length of tweets (100k up to 1m tweets) have been created and provided here under the [/input](https://github.com/Coursal/Text-Sentiment-Analysis-In-Hadoop-And-Spark/tree/master/input) directory named `train#` and `test#` (`#` being a number _1-10_).
 
For the Spark-based applications, 10 datasets that differ in length of tweets (100k up to 1m tweets) have been created and provided here under the [/input](https://github.com/Coursal/Text-Sentiment-Analysis-In-Hadoop-And-Spark/tree/master/input) directory appropriately named `spark_input_#` (`#` being a number _1-10_).
 
## Execution Guide
Up to the development of the source code and compiling of the thesis, the most current stable releases were:
* **Apache Hadoop 3.3.0**
* **Apache Spark 3.0.1**

#### Hadoop-based Application Execution Guide
##### Simple Version of Naive Bayes
```
javac -classpath "$(yarn classpath)" -d NB_classes NB.java
jar -cvf NB.jar -C NB_classes/ .
hadoop jar NB.jar NB train# test# training_split testing_split
```

##### Modified Version of Naive Bayes
```
javac -classpath "$(yarn classpath)" -d Modified_NB_classes Modified_NB.java
jar -cvf Modified_NB.jar -C Modified_NB_classes/ .
hadoop jar Modified_NB.jar Modified_NB train# test# training_split testing_split
```

Where `train#`, `test#` are the desires datasets to be used from [/input](https://github.com/Coursal/Text-Sentiment-Analysis-In-Hadoop-And-Spark/tree/master/input) and `training_split`, `testing_split` are the amounts of chunks to be split in a number of mappers (must be defined as bytes).

---

#### Spark-based Application Execution Guide
##### Simple Version of Naive Bayes
```
sbt package
spark-submit --master yarn --deploy-mode client ./target/scala-2.12/nb_2.12-0.1.jar #
```

##### Modified Version of Naive Bayes
```
sbt package
spark-submit --master yarn --deploy-mode client ./target/scala-2.12/modified_nb_2.12-0.1.jar #
```

##### Simple Version of SVM
```
sbt package
spark-submit --master yarn --deploy-mode client ./target/scala-2.12/svm_2.12-0.1.jar #
```

##### Modified Version of SVM
```
sbt package
spark-submit --master yarn --deploy-mode client ./target/scala-2.12/modified_svm_2.12-0.1.jar #
```

Where `#` is the number indicating the desired dataset to be used from [/input](https://github.com/Coursal/Text-Sentiment-Analysis-In-Hadoop-And-Spark/tree/master/input).
