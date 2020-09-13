import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.Counters;

import java.io.*;
import java.io.IOException;
import java.util.*;
import java.nio.charset.StandardCharsets;

/*
Execution Guide:
hadoop com.sun.tools.javac.Main NB_TFIDF.java
jar cf NB_TFIDF.jar NB_TFIDF*.class
hadoop jar NB_TFIDF.jar NB_TFIDF
hadoop fs -cat output/part-r-00000
*/

public class NB_TFIDF
{
    public static enum Global_Counters 
    {
        NUM_OF_TWEETS, 
        NUM_OF_FEATURES,
        TWEETS_SIZE,
        POS_TWEETS_SIZE,
        NEG_TWEETS_SIZE,
        POS_WORDS_SIZE,
        NEG_WORDS_SIZE,
        FEATURES_SIZE,
        TRUE_POSITIVE_RATE,
        FALSE_POSITIVE_RATE,
        TRUE_NEGATIVE_RATE,
        FALSE_NEGATIVE_RATE
    }



    /* input:  <byte_offset, line_of_tweet>
     * output: <(word@tweet), 1>
     */
    public static class Map_WordCount extends Mapper<Object, Text, Text, IntWritable> 
    {
        private Text word_tweet_key = new Text();
        private final static IntWritable one = new IntWritable(1);

        // string that will hold every tweet IDs with positive sentiment in order to determine the tweets with positive
        // and negative sentiment after the feature selection
        String positive_tweets_id_list = "";    
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException 
        {
            context.getCounter(Global_Counters.NUM_OF_TWEETS).increment(1);

            String line = value.toString();
            String[] columns = line.split(",");

            // if the columns are more than 4, that means the text of the post had commas inside,  
            // so stitch the last columns together to form the full text of the tweet
            if(columns.length > 4)
            {
                for(int i=4; i<columns.length; i++)
                    columns[3] += columns[i];
            }

            String tweet_id = columns[0];
            String tweet_sentiment = columns[1];
            String tweet_text = columns[3];

            if(tweet_sentiment.equals("1"))
                positive_tweets_id_list += tweet_id + "*";  // using the '*' character as a delimiter between tweet IDs


            // clean the text of the tweet from links..
            tweet_text = tweet_text.replaceAll("(http|https)\\:\\/\\/[a-zA-Z0-9\\-\\.]+\\.[a-zA-Z]{2,3}(\\/\\S*)?", "")
                                .replaceAll("#|@|&.*?\\s", "")  // mentions, hashtags, special characters...
                                .replaceAll("\\d+", "")         // numbers...
                                .replaceAll("[^a-zA-Z ]", "")   // punctuation...
                                .toLowerCase()                  // turn every character left to lowercase...
                                .trim()                         // trim the spaces before & after the whole string...
                                .replaceAll("\\s+", " ");       // and get rid of double spaces


            if(tweet_text != null && !tweet_text.trim().isEmpty())
            {
                String[] tweet_words = tweet_text.split(" ");

                for(int i=0; i<tweet_words.length; i++)
                {
                    word_tweet_key.set(tweet_words[i] + "@" + tweet_id);

                    context.write(word_tweet_key, one);
                }
            }
        }

        protected void cleanup(Context context) throws IOException, InterruptedException 
        {
            // create a file in the HDFS to store the tweet IDs with positive sentiment
            FileSystem fs = FileSystem.get(context.getConfiguration());
            Path path = new Path("positive_tweets_id_list");
            FSDataOutputStream os = fs.create(path);
            os.write(positive_tweets_id_list.getBytes(StandardCharsets.UTF_8));
            os.close();
        }
    }

    /* input:  <(word@tweet), 1>
     * output: <(word@tweet), word_count>
     */
    public static class Reduce_WordCount extends Reducer<Text, IntWritable, Text, IntWritable> 
    {
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException 
        {
            int sum = 0;
            for(IntWritable value : values)
                sum += value.get();

            context.write(key, new IntWritable(sum));
        }
    }



    /* input:  <(word@tweet), word_count>
     * output: <tweet, (word=word_count)>
     */
    public static class Map_TF extends Mapper<Object, Text, Text, Text> 
    {
        private Text tweet_key = new Text();
        private Text word_wordcount_value = new Text();
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException 
        {
            String lines[] = value.toString().split("\t");
            String splitted_key[] = lines[0].toString().split("@");

            tweet_key.set(splitted_key[1]);
            word_wordcount_value.set(splitted_key[0] + "=" + lines[1]);

            context.write(tweet_key, word_wordcount_value);    
        }
    }

    /* input:  <tweet, (word=word_count)>
     * output: <(word@tweet), (word_count/tweet_length)>
     */
    public static class Reduce_TF extends Reducer<Text, Text, Text, Text> 
    {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException 
        {
            String tweet = key.toString();

            int total = 0;

            ArrayList<String> word_list = new ArrayList<String>();
            ArrayList<String> count_list = new ArrayList<String>();
            ArrayList<String> word_by_count_list = new ArrayList<String>();

            for(Text value : values)
            {
                String[] splitted_key = value.toString().split("=");

                word_list.add(splitted_key[0]);
                count_list.add(splitted_key[1]);

                total += Integer.parseInt(splitted_key[1]);
            } 
            
            // create the count/total in the list
            for(String count : count_list)
                word_by_count_list.add(count + "/" + total);
        
            // create the iterator for word_list and word_by_count_list and write the key-value pair in the context
            Iterator<String> wl = word_list.iterator();
            Iterator<String> wbcl = word_by_count_list.iterator();
            
            while (wl.hasNext() && wbcl.hasNext()) 
            {               
                context.write(new Text(wl.next() + "@" + tweet), new Text(wbcl.next().toString()));
            }
        }
    }



    /* input:  <(word@tweet), (word_count/tweet_length)>
     * output: <word, (tweet=word_count/tweet_length)>
     */
    public static class Map_TFIDF extends Mapper<Object, Text, Text, Text> 
    {
        private Text word_key = new Text();
        private Text tweet_wordcount_tweetsize_value = new Text();

        public void map(Object key, Text value, Context context ) throws IOException, InterruptedException 
        {
            String[] columns = value.toString().split("\t");
            String[] splitted_key = columns[0].toString().split("@");

            word_key.set(splitted_key[0]);
            tweet_wordcount_tweetsize_value.set(splitted_key[1] + "=" + columns[1]);

            context.write(word_key, tweet_wordcount_tweetsize_value);          
        }
    }

    /* input:  <word, (tweet=word_count/tweet_length)>
     * output: <(tweet@word), TFIDF>
     */
    public static class Reduce_TFIDF extends Reducer<Text, Text, Text, Text> 
    {
        private static int num_of_tweets;
        private Double tfidf;
        
        protected void setup(Context context) throws IOException, InterruptedException 
        {
            Configuration conf = context.getConfiguration();
            num_of_tweets = Integer.parseInt(conf.get("num_of_tweets"));
        }
        
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException 
        {
            int num_of_tweets_with_this_word = 0;
            ArrayList<String> value_list = new ArrayList<String>();

            for (Text value : values)
            {
                value_list.add(value.toString());
                num_of_tweets_with_this_word++;
            }

            // access the list created above, calculate the TFIDF for each word and arrange the values in the required format
            for (String value : value_list)
            {
                String[] value_arr = value.split("=");
                String[] divide_data = value_arr[1].toString().split("/");
                
                tfidf = (Double.parseDouble(divide_data[0])/Double.parseDouble(divide_data[1])) * Math.log(num_of_tweets/num_of_tweets_with_this_word);
                
                context.write(new Text(value_arr[0] + "@" + key.toString()), new Text(tfidf.toString()));

                context.getCounter(Global_Counters.NUM_OF_FEATURES).increment(1);
            }           
        }
    }



    /* input:  <(tweet@word), TFIDF>
     * output: <NULL, (tweet@word_TFIDF)>
     */
    public static class Map_FeatSel extends Mapper<Object, Text, NullWritable, Text>
    {
        private int features_to_keep;
        private java.util.Map<Text, Double> top_features = new HashMap<Text, Double>();

        protected void setup(Context context) throws IOException, InterruptedException 
        {
            features_to_keep = Integer.parseInt(context.getConfiguration().get("features_to_keep"));
        }
                
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException
        {
            String[] columns = value.toString().split("\t");
            
            top_features.put(new Text(columns[0]), Double.parseDouble(columns[1])); 

            // sort the words by their TFIDF score
            List list = new LinkedList(top_features.entrySet());
            Collections.sort(list, new Comparator()
                                    {
                                        public int compare(Object o1, Object o2) 
                                        {
                                            return ((Comparable) ((java.util.Map.Entry) (o1))
                                                    .getValue()).compareTo(((java.util.Map.Entry) (o2))
                                                    .getValue());
                                        }
                                    });


            HashMap sorted_top_features = new LinkedHashMap();
            for(Iterator i = list.iterator(); i.hasNext();)
            {
                java.util.Map.Entry entry = (java.util.Map.Entry) i.next();
                sorted_top_features.put(entry.getKey(), entry.getValue());
            }

            // hold the words with the biggest TFIDF score
            while(sorted_top_features.size() > features_to_keep)
                sorted_top_features.remove(sorted_top_features.keySet().stream().findFirst().get());

            Set set = sorted_top_features.entrySet();
            Iterator it = set.iterator();
            while(it.hasNext())
            {
                java.util.Map.Entry me = (java.util.Map.Entry) it.next();
                context.write(NullWritable.get(), new Text(me.getKey().toString() + "_" + String.valueOf(me.getValue())));
            }
        }
    }

    /* input:  <NULL, tweet@word_TFIDF>
     * output: <TFIDF, tweet@word>
     */
    public static class Reduce_FeatSel extends Reducer<NullWritable, Text, Text, Text>
    {
        private int features_to_keep;
        private java.util.Map<Text, Double> top_features = new HashMap<Text, Double>();

        protected void setup(Context context) throws IOException, InterruptedException 
        {
            features_to_keep = Integer.parseInt(context.getConfiguration().get("features_to_keep"));
        }
        
        public void reduce(NullWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException 
        {
            for(Text value : values) 
            {       
                String[] columns = value.toString().split("_");
                top_features.put(new Text(columns[0]), Double.parseDouble(columns[1]));     
            } 

            // sort the words by their TFIDF score
            List list = new LinkedList(top_features.entrySet());
            Collections.sort(list, new Comparator()
                                    {
                                        public int compare(Object o1, Object o2) 
                                        {
                                            return ((Comparable) ((java.util.Map.Entry) (o1))
                                                    .getValue()).compareTo(((java.util.Map.Entry) (o2))
                                                    .getValue());
                                        }
                                    });

            HashMap sorted_top_features = new LinkedHashMap();
            for(Iterator i = list.iterator(); i.hasNext();)
            {
                java.util.Map.Entry entry = (java.util.Map.Entry) i.next();
                sorted_top_features.put(entry.getKey(), entry.getValue());
            }

            // hold the words with the biggest TFIDF score
            while(sorted_top_features.size() > features_to_keep)
                sorted_top_features.remove(sorted_top_features.keySet().stream().findFirst().get());

            Set set = sorted_top_features.entrySet();
            Iterator it = set.iterator();

            while(it.hasNext())
            {
                java.util.Map.Entry me = (java.util.Map.Entry) it.next();
                context.write(new Text(String.valueOf(me.getValue())), new Text(me.getKey().toString()));
            }
        }
    }



    /* input:  <TFIDF, tweet@word>
     * output: <tweet, word>
     */
    public static class Map_Training_1 extends Mapper<Object, Text, Text, Text> 
    {
        private Text tweet_key = new Text();
        private Text word_value = new Text();
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException 
        {
            String line[] = value.toString().split("\t");
            String splitted_value[] = line[1].toString().split("@");

            tweet_key.set(splitted_value[0]);
            word_value.set(splitted_value[1]);

            context.write(tweet_key, word_value);      
        }
    }

    /* input:  <tweet, word>
     * output: <tweet, (tweet_text#sentiment)>
     */
    public static class Reduce_Training_1 extends Reducer<Text, Text, Text, Text> 
    {
        private String positive_tweets_IDs;
        private String[] positive_tweets_IDs_arr;
        
        protected void setup(Context context) throws IOException, InterruptedException 
        {
            positive_tweets_IDs = context.getConfiguration().get("positive_tweets_IDs");
            positive_tweets_IDs_arr = positive_tweets_IDs.split("\\*");
        }

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException 
        {
            String tweet = key.toString();

            context.getCounter(Global_Counters.TWEETS_SIZE).increment(1);

            String word_list = "";

            boolean flag = false;
            String sentiment = "NEGATIVE";

            for(String tweet_id : positive_tweets_IDs_arr)
            {
                if(tweet.equals(tweet_id))
                    flag = true;
            }

            if(flag)
            {
                sentiment = "POSITIVE";
                context.getCounter(Global_Counters.POS_TWEETS_SIZE).increment(1);
            }
            else
                context.getCounter(Global_Counters.NEG_TWEETS_SIZE).increment(1);


            for(Text value : values)
            {
                String word = value.toString();


                word_list += word + " ";

                if(sentiment.equals("POSITIVE"))
                    context.getCounter(Global_Counters.POS_WORDS_SIZE).increment(1);
                else
                    context.getCounter(Global_Counters.NEG_WORDS_SIZE).increment(1);
            } 
            
            
            context.write(key, new Text(word_list + "#" + sentiment));
        }
    }



    /* input:  <tweet, (tweet_text#sentiment)>
     * output: <word, sentiment>
     */
    public static class Map_Training_2 extends Mapper<Object, Text, Text, Text> 
    {
        private Text word_key = new Text();
        private Text sentiment_value = new Text();
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException 
        {
            String[] lines = value.toString().split("\t");
            String[] splitted_value = lines[1].toString().split("#");
            String[] tweet_words = splitted_value[0].split(" ");

            for(String word : tweet_words)
            {
                word_key.set(word);
                sentiment_value.set(splitted_value[1]);

                context.write(word_key, sentiment_value);
            }  
        }
    }

    /* input:  <word, sentiment>
     * output: <word, pos_wordcount@neg_wordcount>
     */
    public static class Reduce_Training_2 extends Reducer<Text, Text, Text, Text> 
    {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException 
        {
            context.getCounter(Global_Counters.FEATURES_SIZE).increment(1); 

            int positive_counter = 0;
            int negative_counter = 0;

            // for each word, count the occurences in tweets with positive/negative sentiment
            for(Text value : values)
            {
                String sentiment = value.toString();
                if(sentiment.equals("POSITIVE"))
                    positive_counter++;
                else
                    negative_counter++;
            }

            context.write(key, new Text(String.valueOf(positive_counter) + "@" + String.valueOf(negative_counter)));    
        }
    }



    /* input: <byte_offset, line_of_tweet>
     * output: <tweet@tweet_text, sentiment>
     */
    public static class Map_Testing extends Mapper<Object, Text, Text, Text> 
    {
        int features_size, tweets_size, pos_tweets_size, neg_tweets_size, pos_words_size, neg_words_size;
        Double pos_class_probability, neg_class_probability;

        // hashmaps with each word as key and its number of occurences in each class as value
        HashMap<String, Integer> pos_words = new HashMap<String, Integer>();
        HashMap<String, Integer> neg_words = new HashMap<String, Integer>();

        // hashmaps with each word as key and its probability in each class as value
        HashMap<String, Double> pos_words_probabilities = new HashMap<String, Double>();
        HashMap<String, Double> neg_words_probabilities = new HashMap<String, Double>();

        // lists holding all probabilities to be multiplied together, along with the positive/negative class probability
        ArrayList<Double> pos_probabilities_list = new ArrayList<Double>();
        ArrayList<Double> neg_probabilities_list = new ArrayList<Double>();

        protected void setup(Context context) throws IOException, InterruptedException 
        {
            // load all counters to be used for the calculation of the probabilities
            features_size = Integer.parseInt(context.getConfiguration().get("features_size"));
            tweets_size = Integer.parseInt(context.getConfiguration().get("tweets_size"));
            pos_tweets_size = Integer.parseInt(context.getConfiguration().get("pos_tweets_size"));
            neg_tweets_size = Integer.parseInt(context.getConfiguration().get("neg_tweets_size"));
            pos_words_size = Integer.parseInt(context.getConfiguration().get("pos_words_size"));
            neg_words_size = Integer.parseInt(context.getConfiguration().get("neg_words_size"));

            pos_class_probability = ((double) pos_tweets_size) / tweets_size;
            neg_class_probability = ((double) neg_tweets_size) / tweets_size;

            // load the model of the last training job and fill two hashmaps of words with the number of
            // occurences in positive and negative tweets
            Path training_model = new Path("training_2");
            FileSystem model_fs = training_model.getFileSystem(context.getConfiguration());
            FileStatus[] file_status = model_fs.listStatus(training_model);

            for(FileStatus i : file_status)
            {
                Path current_file_path = i.getPath();

                if(i.isFile())
                {
                    BufferedReader br = new BufferedReader(new InputStreamReader(model_fs.open(current_file_path)));
                    String line; 

                    while((line = br.readLine()) != null)
                    {
                        String[] columns = line.toString().split("\t");
                        String[] pos_and_neg_counts = columns[1].split("@");

                        pos_words.put(columns[0], Integer.parseInt(pos_and_neg_counts[0]));
                        neg_words.put(columns[0], Integer.parseInt(pos_and_neg_counts[1]));
                    }

                    br.close();
                }
            }

            // calculate all the word probabilities for positive and negative class (with laplace smoothing)
            for(Map.Entry<String,Integer> entry : pos_words.entrySet()) 
            {
                pos_words_probabilities.put(entry.getKey(), ((double) entry.getValue() + 1) / (pos_words_size + features_size));
                neg_words_probabilities.put(entry.getKey(), ((double) neg_words.get(entry.getKey()) + 1) / (neg_words_size + features_size));
            }
        }
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException 
        {
            String line = value.toString();
            String[] columns = line.split(",");

            // if the columns are more than 4, that means the text of the post had commas inside,  
            // so stitch the last columns together to form the post
            if(columns.length > 4)
            {
                for(int i=4; i<columns.length; i++)
                    columns[3] += columns[i];
            }

            String tweet_id = columns[0];
            String tweet_sentiment = columns[1];
            String tweet_text = columns[3];

            // clean the text of the tweet from links..
            tweet_text = tweet_text.replaceAll("(http|https)\\:\\/\\/[a-zA-Z0-9\\-\\.]+\\.[a-zA-Z]{2,3}(\\/\\S*)?", "")
                                .replaceAll("#|@|&.*?\\s", "")  // mentions, hashtags, special characters...
                                .replaceAll("\\d+", "")         // numbers...
                                .replaceAll("[^a-zA-Z ]", "")   // punctuation...
                                .toLowerCase()                  // turn every character left to lowercase...
                                .trim()                         // trim the spaces before & after the whole string...
                                .replaceAll("\\s+", " ");       // and get rid of double spaces

            // initialize the probabilities with the class probability of each sentiment
            Double pos_probability = pos_class_probability;
            Double neg_probability = neg_class_probability;

            // calculate the product of the probabilities of the words (+ the class probability) for each class
            if(tweet_text != null && !tweet_text.trim().isEmpty())
            {
                String[] tweet_words = tweet_text.split(" ");

                for(String word : tweet_words)
                {
                    for(Map.Entry<String,Double> entry : pos_words_probabilities.entrySet()) 
                    {
                        if(word.equals(entry.getKey()))
                        {
                            pos_probability = ((double) pos_probability) * pos_words_probabilities.get(word);
                            neg_probability = ((double) neg_probability) * neg_words_probabilities.get(word);
                        }
                    }
                }
            }

            // compare and set the max value of the two class probabilities as the result of the guessed sentiment for every tweet
            if(Double.compare(pos_probability, neg_probability) > 0)
            {
                if(tweet_sentiment.equals("1"))
                    context.getCounter(Global_Counters.TRUE_POSITIVE_RATE).increment(1);
                else
                    context.getCounter(Global_Counters.FALSE_POSITIVE_RATE).increment(1);

                context.write(new Text(tweet_id + "@" + tweet_text), new Text("POSITIVE"));
            }
            else
            {
                if(tweet_sentiment.equals("0"))
                    context.getCounter(Global_Counters.TRUE_NEGATIVE_RATE).increment(1);
                else
                    context.getCounter(Global_Counters.FALSE_NEGATIVE_RATE).increment(1);

                context.write(new Text(tweet_id + "@" + tweet_text), new Text("NEGATIVE"));
            }
        }
    }



    public static void main(String[] args) throws Exception 
    {
        // paths to directories were inbetween and final job outputs are stored
        Path input_dir = new Path("input");
        Path wordcount_dir = new Path("wordcount");
        Path tf_dir = new Path("tf");
        Path tfidf_dir = new Path("tfidf");
        Path features_dir = new Path("features");
        Path training_1_dir = new Path("training_1");
        Path training_2_dir = new Path("training_2");
        Path testing_dir = new Path("test_data");
        Path output_dir = new Path("output");

        Path positive_tweets_id_list = new Path("positive_tweets_id_list"); // file with the tweet IDs with positive sentiment

        Configuration conf = new Configuration();

        FileSystem fs = FileSystem.get(conf);
        if(fs.exists(wordcount_dir))
            fs.delete(wordcount_dir, true);
        if(fs.exists(tf_dir))
            fs.delete(tf_dir, true);
        if(fs.exists(tfidf_dir))
            fs.delete(tfidf_dir, true);
        if(fs.exists(features_dir))
            fs.delete(features_dir, true);
        if(fs.exists(training_1_dir))
            fs.delete(training_1_dir, true);
        if(fs.exists(training_2_dir))
            fs.delete(training_2_dir, true);
        if(fs.exists(output_dir))
            fs.delete(output_dir, true);
        if(fs.exists(positive_tweets_id_list))
            fs.delete(positive_tweets_id_list, true);

        long start_time = System.nanoTime();

        Job wordcount_job = Job.getInstance(conf, "Word Count");
        wordcount_job.setJarByClass(NB_TFIDF.class);
        wordcount_job.setMapperClass(Map_WordCount.class);
        wordcount_job.setCombinerClass(Reduce_WordCount.class);
        wordcount_job.setReducerClass(Reduce_WordCount.class);
        wordcount_job.setMapOutputKeyClass(Text.class);
        wordcount_job.setMapOutputValueClass(IntWritable.class);
        wordcount_job.setOutputKeyClass(Text.class);
        wordcount_job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(wordcount_job, input_dir);
        FileOutputFormat.setOutputPath(wordcount_job, wordcount_dir);
        wordcount_job.waitForCompletion(true);


        // read the file with the tweet IDs with positive sentiment
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(positive_tweets_id_list)));
        String line = br.readLine();
        br.close();
        conf.set("positive_tweets_IDs", line);


        // Counting the total number of tweets from the training data in order to calculate the TFIDF score of each feature
        int num_of_tweets = Math.toIntExact(wordcount_job.getCounters().findCounter(Global_Counters.NUM_OF_TWEETS).getValue());
        conf.set("num_of_tweets", String.valueOf(num_of_tweets));
        System.out.println("TOTAL TWEET NUMBER: " + num_of_tweets);


        Job tf_job = Job.getInstance(conf, "TF");
        tf_job.setJarByClass(NB_TFIDF.class);
        tf_job.setMapperClass(Map_TF.class);
        tf_job.setReducerClass(Reduce_TF.class);
        tf_job.setMapOutputKeyClass(Text.class);
        tf_job.setMapOutputValueClass(Text.class);
        tf_job.setOutputKeyClass(Text.class);
        tf_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(tf_job, wordcount_dir);
        FileOutputFormat.setOutputPath(tf_job, tf_dir);
        tf_job.waitForCompletion(true);

        Job tfidf_job = Job.getInstance(conf, "TFIDF");
        tfidf_job.setJarByClass(NB_TFIDF.class);
        tfidf_job.setMapperClass(Map_TFIDF.class);
        tfidf_job.setReducerClass(Reduce_TFIDF.class);
        tfidf_job.setMapOutputKeyClass(Text.class);
        tfidf_job.setMapOutputValueClass(Text.class);
        tfidf_job.setOutputKeyClass(Text.class);
        tfidf_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(tfidf_job, tf_dir);
        FileOutputFormat.setOutputPath(tfidf_job, tfidf_dir);
        tfidf_job.waitForCompletion(true);


        // Calculating the total number of words that got vectorized in order to find out how many words/features to keep, just 
        // so the model has to work only with the 20% of the most relevant of the features, based on the Pareto principle
        int num_of_features = Math.toIntExact(tfidf_job.getCounters().findCounter(Global_Counters.NUM_OF_FEATURES).getValue());
        System.out.println("TOTAL NUMBER OF FEATURES: " + num_of_features);
        int features_to_keep = (num_of_features * 20) / 100;
        conf.set("features_to_keep", String.valueOf(features_to_keep));
        System.out.println("FEATURES TO KEEP: " + features_to_keep);


        Job feature_selection_job = Job.getInstance(conf, "Feature Selection");
        feature_selection_job.setJarByClass(NB_TFIDF.class);
        feature_selection_job.setMapperClass(Map_FeatSel.class);
        feature_selection_job.setReducerClass(Reduce_FeatSel.class);
        feature_selection_job.setNumReduceTasks(1); 
        feature_selection_job.setMapOutputKeyClass(NullWritable.class);
        feature_selection_job.setMapOutputValueClass(Text.class);
        feature_selection_job.setOutputKeyClass(Text.class);
        feature_selection_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(feature_selection_job, tfidf_dir);
        FileOutputFormat.setOutputPath(feature_selection_job, features_dir);
        feature_selection_job.waitForCompletion(true);

        Job training_1_job = Job.getInstance(conf, "Training Part 1");
        training_1_job.setJarByClass(NB_TFIDF.class);
        training_1_job.setMapperClass(Map_Training_1.class);
        training_1_job.setReducerClass(Reduce_Training_1.class);    
        training_1_job.setMapOutputKeyClass(Text.class);
        training_1_job.setMapOutputValueClass(Text.class);
        training_1_job.setOutputKeyClass(Text.class);
        training_1_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(training_1_job, features_dir);
        FileOutputFormat.setOutputPath(training_1_job, training_1_dir);
        training_1_job.waitForCompletion(true);

        int tweets_size = Math.toIntExact(training_1_job.getCounters().findCounter(Global_Counters.TWEETS_SIZE).getValue());
        conf.set("tweets_size", String.valueOf(tweets_size));
        int pos_tweets_size = Math.toIntExact(training_1_job.getCounters().findCounter(Global_Counters.POS_TWEETS_SIZE).getValue());
        conf.set("pos_tweets_size", String.valueOf(pos_tweets_size));
        int neg_tweets_size = Math.toIntExact(training_1_job.getCounters().findCounter(Global_Counters.NEG_TWEETS_SIZE).getValue());
        conf.set("neg_tweets_size", String.valueOf(neg_tweets_size));
        int pos_words_size = Math.toIntExact(training_1_job.getCounters().findCounter(Global_Counters.POS_WORDS_SIZE).getValue());
        conf.set("pos_words_size", String.valueOf(pos_words_size));
        int neg_words_size = Math.toIntExact(training_1_job.getCounters().findCounter(Global_Counters.NEG_WORDS_SIZE).getValue());
        conf.set("neg_words_size", String.valueOf(neg_words_size));

        Job training_2_job = Job.getInstance(conf, "Training Part 2");
        training_2_job.setJarByClass(NB_TFIDF.class);
        training_2_job.setMapperClass(Map_Training_2.class);
        training_2_job.setReducerClass(Reduce_Training_2.class);    
        training_2_job.setMapOutputKeyClass(Text.class);
        training_2_job.setMapOutputValueClass(Text.class);
        training_2_job.setOutputKeyClass(Text.class);
        training_2_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(training_2_job, training_1_dir);
        FileOutputFormat.setOutputPath(training_2_job, training_2_dir);
        training_2_job.waitForCompletion(true);

        int features_size = Math.toIntExact(training_2_job.getCounters().findCounter(Global_Counters.FEATURES_SIZE).getValue());
        conf.set("features_size", String.valueOf(features_size));

        Job testing_job = Job.getInstance(conf, "Testing");
        testing_job.setJarByClass(NB_TFIDF.class);
        testing_job.setMapperClass(Map_Testing.class);  
        testing_job.setMapOutputKeyClass(Text.class);
        testing_job.setMapOutputValueClass(Text.class);
        testing_job.setOutputKeyClass(Text.class);
        testing_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(testing_job, testing_dir);
        FileOutputFormat.setOutputPath(testing_job, output_dir);
        testing_job.waitForCompletion(true);

        System.out.println("EXECUTION DURATION: " + (System.nanoTime() - start_time) / 1000000000F + " seconds");

        int tp = Math.toIntExact(testing_job.getCounters().findCounter(Global_Counters.TRUE_POSITIVE_RATE).getValue());
        int fp = Math.toIntExact(testing_job.getCounters().findCounter(Global_Counters.FALSE_POSITIVE_RATE).getValue());
        int tn = Math.toIntExact(testing_job.getCounters().findCounter(Global_Counters.TRUE_NEGATIVE_RATE).getValue());
        int fn = Math.toIntExact(testing_job.getCounters().findCounter(Global_Counters.FALSE_NEGATIVE_RATE).getValue());

        System.out.println("\nCONFUSION MATRIX:");
        System.out.printf("%-10s %-10s \n", tp, fp);
        System.out.printf("%-10s %-10s \n\n", fn, tn);

        System.out.printf("%-25s %-10s \n", "SENSITIVITY: ", ((double) tp) / (tp + fn));
        System.out.printf("%-25s %-10s \n", "PRECISION: ", ((double) tp) / (tp + fp));
        System.out.printf("%-25s %-10s \n", "ACCURACY: ", ((double) (tp + tn)) / (tp + tn + fp + fn));
        System.out.printf("%-25s %-10s \n", "BALANCED ACCURACY: ", ((double) (((double) tp) / (tp + fn) + ((double) tn) / (tn + fp))) / 2);
        System.out.printf("%-25s %-10s \n", "F1 SCORE: ", ((double) (2 * tp)) / (2 * tp + fp + fn));
    }
}