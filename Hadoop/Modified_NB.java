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
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
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


public class Modified_NB
{
    public static enum Global_Counters 
    {
        NUM_OF_TWEETS, 
        TWEETS_SIZE,
        POS_TWEETS_SIZE,
        NEG_TWEETS_SIZE,
        POS_WORDS_SIZE,
        NEG_WORDS_SIZE,
        FEATURES_SIZE,
        TRUE_POSITIVE,
        FALSE_POSITIVE,
        TRUE_NEGATIVE,
        FALSE_NEGATIVE
    }



    /* input:  <byte_offset, line_of_tweet>
     * output: <(word@tweet), 1>
     */
    public static class Map_WordCount extends Mapper<Object, Text, Text, IntWritable> 
    {
        private Text word_tweet_key = new Text();
        private final static IntWritable one = new IntWritable(1);

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
                tweet_id += '+';

            // clean the text of the tweet from links...
            tweet_text = tweet_text.replaceAll("(?i)(https?:\\/\\/(?:www\\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\\.[^\\s]{2,}|www\\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\\.[^\\s]{2,}|https?:\\/\\/(?:www\\.|(?!www))[a-zA-Z0-9]+\\.[^\\s]{2,}|www\\.[a-zA-Z0-9]+\\.[^\\s]{2,})", "")
                                .replaceAll("(#|@|&).*?\\w+", "")   // mentions, hashtags, special characters...
                                .replaceAll("\\d+", "")             // numbers...
                                .replaceAll("[^a-zA-Z ]", " ")      // punctuation...
                                .toLowerCase()                      // turn every character left to lowercase...
                                .trim()                             // trim the spaces before & after the whole string...
                                .replaceAll("\\s+", " ");           // and get rid of double spaces


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
            }         
        }
    }



    /* input:  <(tweet@word), TFIDF>
     * output: <tweet, (word_TFIDF)>
     */
    public static class Map_FeatSel extends Mapper<Object, Text, Text, Text>
    {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException
        {
            String[] columns = value.toString().split("\t");

            String[] splitted_key = columns[0].toString().split("@");

            context.write(new Text(splitted_key[0]), new Text(splitted_key[1] + "_" + columns[1]));
        }
    }

    /* input:  <tweet, (word_TFIDF)>
     * output: <tweet, text>
     */
    public static class Reduce_FeatSel extends Reducer<Text, Text, Text, Text>
    {
        private HashMap tweet_words = new HashMap<Text, Double>();

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException 
        {
            context.getCounter(Global_Counters.TWEETS_SIZE).increment(1);

            String tweet_text = "";

            for(Text value : values) 
            {
                String[] columns = value.toString().split("_");
                tweet_words.put(new Text(columns[0]), Double.parseDouble(columns[1]));    
            }

            // sort the tweet's words by their TFIDF score
            List list = new LinkedList(tweet_words.entrySet());

            int num_of_words = tweet_words.size();
            tweet_words.clear();

            Collections.sort(list, new Comparator()
                                    {
                                        public int compare(Object o1, Object o2) 
                                        {
                                            return ((Comparable) ((java.util.Map.Entry) (o1)).getValue())
                                                    .compareTo(((java.util.Map.Entry) (o2)).getValue());
                                        }
                                    });

            HashMap sorted_tweet_words = new LinkedHashMap();
            for(Iterator i = list.iterator(); i.hasNext();)
            {
                java.util.Map.Entry entry = (java.util.Map.Entry) i.next();
                sorted_tweet_words.put(entry.getKey(), entry.getValue());
            }

            // hold the words with the biggest TFIDF score, by trimming down the ones with the lowest score 
            // until the 75% of the words remained in the list
            while((sorted_tweet_words.size() > ((num_of_words * 75) / 100)) && (num_of_words > 1))
                sorted_tweet_words.remove(sorted_tweet_words.keySet().stream().findFirst().get());

            if(key.toString().endsWith("+"))
            {
                context.getCounter(Global_Counters.POS_TWEETS_SIZE).increment(1);
                context.getCounter(Global_Counters.POS_WORDS_SIZE).increment(sorted_tweet_words.size());
            }
            else
            {
                context.getCounter(Global_Counters.NEG_TWEETS_SIZE).increment(1);
                context.getCounter(Global_Counters.NEG_WORDS_SIZE).increment(sorted_tweet_words.size());
            }

            // put the most relevant words in a string
            Set set = sorted_tweet_words.entrySet();
            Iterator it = set.iterator();
            while(it.hasNext())
            {
                java.util.Map.Entry me = (java.util.Map.Entry) it.next();
                tweet_text += me.getKey().toString() + " ";
            }

            context.write(key, new Text(tweet_text));
        }
    }



    /* input:  <tweet, text>
     * output: <word, sentiment>
     */
    public static class Map_Training extends Mapper<Object, Text, Text, Text> 
    {
        private Text word_key = new Text();
        private Text sentiment_value = new Text();
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException 
        {
            String[] line = value.toString().split("\t");
            String[] tweet_words = line[1].toString().split(" ");

            if(line[0].endsWith("+"))
                sentiment_value.set("POSITIVE");
            else
                sentiment_value.set("NEGATIVE");

            for(String word : tweet_words)
            {
                word_key.set(word);

                context.write(word_key, sentiment_value);
            }  
        }
    }

    /* input:  <word, sentiment>
     * output: <word, pos_wordcount@neg_wordcount>
     */
    public static class Reduce_Training extends Reducer<Text, Text, Text, Text> 
    {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException 
        {
            context.getCounter(Global_Counters.FEATURES_SIZE).increment(1); 

            int positive_counter = 0;
            int negative_counter = 0;

            // for each word, count the occurrences in tweets with positive/negative sentiment
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

        // hashmaps with each word as key and its number of occurrences in each class as value
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
            // occurrences in positive and negative tweets
            Path training_model = new Path("training");
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
            tweet_text = tweet_text.replaceAll("(?i)(https?:\\/\\/(?:www\\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\\.[^\\s]{2,}|www\\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\\.[^\\s]{2,}|https?:\\/\\/(?:www\\.|(?!www))[a-zA-Z0-9]+\\.[^\\s]{2,}|www\\.[a-zA-Z0-9]+\\.[^\\s]{2,})", "")
                                .replaceAll("(#|@|&).*?\\w+", "")   // mentions, hashtags, special characters...
                                .replaceAll("\\d+", "")             // numbers...
                                .replaceAll("[^a-zA-Z ]", " ")      // punctuation...
                                .toLowerCase()                      // turn every character left to lowercase...
                                .trim()                             // trim the spaces before & after the whole string...
                                .replaceAll("\\s+", " ");           // and get rid of double spaces

            // initialize the product of positive and negative probabilities with 1
            Double pos_probability = 1.0;
            Double neg_probability = 1.0;

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
                            pos_probability *= pos_words_probabilities.get(word);
                            neg_probability *= neg_words_probabilities.get(word);
                        }
                    }
                }
            }

            // multiply the product of positive and negative probability with the class probability of each sentiment
            pos_probability *= pos_class_probability;
            neg_probability *= neg_class_probability;

            // compare and set the max value of the two class probabilities as the result of the guessed sentiment for every tweet
            if(Double.compare(pos_probability, neg_probability) > 0)
            {
                if(tweet_sentiment.equals("1"))
                    context.getCounter(Global_Counters.TRUE_POSITIVE).increment(1);
                else
                    context.getCounter(Global_Counters.FALSE_POSITIVE).increment(1);

                context.write(new Text(tweet_id + "@" + tweet_text), new Text("POSITIVE"));
            }
            else
            {
                if(tweet_sentiment.equals("0"))
                    context.getCounter(Global_Counters.TRUE_NEGATIVE).increment(1);
                else
                    context.getCounter(Global_Counters.FALSE_NEGATIVE).increment(1);

                context.write(new Text(tweet_id + "@" + tweet_text), new Text("NEGATIVE"));
            }
        }
    }



    public static void main(String[] args) throws Exception 
    {
        // paths to directories were input, inbetween and final job outputs are stored
        Path input_dir = new Path(args[0]);
        Path wordcount_dir = new Path("wordcount");
        Path tf_dir = new Path("tf");
        Path tfidf_dir = new Path("tfidf");
        Path features_dir = new Path("features");
        Path training_dir = new Path("training");
        Path testing_dir = new Path(args[1]);
        Path output_dir = new Path("output");

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
        if(fs.exists(training_dir))
            fs.delete(training_dir, true);
        if(fs.exists(output_dir))
            fs.delete(output_dir, true);

        long start_time = System.nanoTime();

        Job wordcount_job = Job.getInstance(conf, "Word Count");
        wordcount_job.setJarByClass(Modified_NB.class);
        wordcount_job.setMapperClass(Map_WordCount.class);
        wordcount_job.setCombinerClass(Reduce_WordCount.class);
        wordcount_job.setReducerClass(Reduce_WordCount.class);
        wordcount_job.setNumReduceTasks(3);
        wordcount_job.setMapOutputKeyClass(Text.class);
        wordcount_job.setMapOutputValueClass(IntWritable.class);
        wordcount_job.setOutputKeyClass(Text.class);
        wordcount_job.setOutputValueClass(IntWritable.class);
        TextInputFormat.addInputPath(wordcount_job, input_dir);
        TextInputFormat.setMaxInputSplitSize(wordcount_job, Long.valueOf(args[2]));
        TextOutputFormat.setOutputPath(wordcount_job, wordcount_dir);
        wordcount_job.waitForCompletion(true);

        // Counting the total number of tweets from the training data in order to calculate the TFIDF score of each feature
        int num_of_tweets = Math.toIntExact(wordcount_job.getCounters().findCounter(Global_Counters.NUM_OF_TWEETS).getValue());
        conf.set("num_of_tweets", String.valueOf(num_of_tweets));

        Job tf_job = Job.getInstance(conf, "TF");
        tf_job.setJarByClass(Modified_NB.class);
        tf_job.setMapperClass(Map_TF.class);
        tf_job.setReducerClass(Reduce_TF.class);
        tf_job.setNumReduceTasks(3);
        tf_job.setMapOutputKeyClass(Text.class);
        tf_job.setMapOutputValueClass(Text.class);
        tf_job.setOutputKeyClass(Text.class);
        tf_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(tf_job, wordcount_dir);
        FileOutputFormat.setOutputPath(tf_job, tf_dir);
        tf_job.waitForCompletion(true);

        Job tfidf_job = Job.getInstance(conf, "TFIDF");
        tfidf_job.setJarByClass(Modified_NB.class);
        tfidf_job.setMapperClass(Map_TFIDF.class);
        tfidf_job.setReducerClass(Reduce_TFIDF.class);
        tfidf_job.setNumReduceTasks(3);
        tfidf_job.setMapOutputKeyClass(Text.class);
        tfidf_job.setMapOutputValueClass(Text.class);
        tfidf_job.setOutputKeyClass(Text.class);
        tfidf_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(tfidf_job, tf_dir);
        FileOutputFormat.setOutputPath(tfidf_job, tfidf_dir);
        tfidf_job.waitForCompletion(true);

        Job feature_selection_job = Job.getInstance(conf, "Feature Selection");
        feature_selection_job.setJarByClass(Modified_NB.class);
        feature_selection_job.setMapperClass(Map_FeatSel.class);
        feature_selection_job.setReducerClass(Reduce_FeatSel.class);
        feature_selection_job.setNumReduceTasks(3);
        feature_selection_job.setMapOutputKeyClass(Text.class);
        feature_selection_job.setMapOutputValueClass(Text.class);
        feature_selection_job.setOutputKeyClass(Text.class);
        feature_selection_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(feature_selection_job, tfidf_dir);
        FileOutputFormat.setOutputPath(feature_selection_job, features_dir);
        feature_selection_job.waitForCompletion(true);

        int tweets_size = Math.toIntExact(feature_selection_job.getCounters().findCounter(Global_Counters.TWEETS_SIZE).getValue());
        conf.set("tweets_size", String.valueOf(tweets_size));
        int pos_tweets_size = Math.toIntExact(feature_selection_job.getCounters().findCounter(Global_Counters.POS_TWEETS_SIZE).getValue());
        conf.set("pos_tweets_size", String.valueOf(pos_tweets_size));
        int neg_tweets_size = Math.toIntExact(feature_selection_job.getCounters().findCounter(Global_Counters.NEG_TWEETS_SIZE).getValue());
        conf.set("neg_tweets_size", String.valueOf(neg_tweets_size));
        int pos_words_size = Math.toIntExact(feature_selection_job.getCounters().findCounter(Global_Counters.POS_WORDS_SIZE).getValue());
        conf.set("pos_words_size", String.valueOf(pos_words_size));
        int neg_words_size = Math.toIntExact(feature_selection_job.getCounters().findCounter(Global_Counters.NEG_WORDS_SIZE).getValue());
        conf.set("neg_words_size", String.valueOf(neg_words_size));

        Job training_job = Job.getInstance(conf, "Training");
        training_job.setJarByClass(Modified_NB.class);
        training_job.setMapperClass(Map_Training.class);
        training_job.setReducerClass(Reduce_Training.class);  
        training_job.setNumReduceTasks(3);   
        training_job.setMapOutputKeyClass(Text.class);
        training_job.setMapOutputValueClass(Text.class);
        training_job.setOutputKeyClass(Text.class);
        training_job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(training_job, features_dir);
        FileOutputFormat.setOutputPath(training_job, training_dir);
        training_job.waitForCompletion(true);

        int features_size = Math.toIntExact(training_job.getCounters().findCounter(Global_Counters.FEATURES_SIZE).getValue());
        conf.set("features_size", String.valueOf(features_size));

        Job testing_job = Job.getInstance(conf, "Testing");
        testing_job.setJarByClass(Modified_NB.class);
        testing_job.setMapperClass(Map_Testing.class);  
        testing_job.setMapOutputKeyClass(Text.class);
        testing_job.setMapOutputValueClass(Text.class);
        testing_job.setOutputKeyClass(Text.class);
        testing_job.setOutputValueClass(Text.class);
        TextInputFormat.addInputPath(testing_job, testing_dir);
        TextInputFormat.setMaxInputSplitSize(testing_job, Long.valueOf(args[3]));
        TextOutputFormat.setOutputPath(testing_job, output_dir);
        testing_job.waitForCompletion(true);

        System.out.println("EXECUTION DURATION: " + (System.nanoTime() - start_time) / 1000000000F + " seconds");

        int tp = Math.toIntExact(testing_job.getCounters().findCounter(Global_Counters.TRUE_POSITIVE).getValue());
        int fp = Math.toIntExact(testing_job.getCounters().findCounter(Global_Counters.FALSE_POSITIVE).getValue());
        int tn = Math.toIntExact(testing_job.getCounters().findCounter(Global_Counters.TRUE_NEGATIVE).getValue());
        int fn = Math.toIntExact(testing_job.getCounters().findCounter(Global_Counters.FALSE_NEGATIVE).getValue());

        System.out.println("\nCONFUSION MATRIX:");
        System.out.printf("%-10s %-10s \n", tp, fp);
        System.out.printf("%-10s %-10s \n\n", fn, tn);

        System.out.printf("%-25s %-10s \n", "ACCURACY: ", ((double) (tp + tn)) / (tp + tn + fp + fn));
    }
}