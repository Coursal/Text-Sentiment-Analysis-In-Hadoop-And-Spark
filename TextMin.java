import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.Counters;

import java.io.*;
import java.io.IOException;
import java.util.*;

/*
Execution Guide:
hadoop com.sun.tools.javac.Main TextMin.java
jar cf TextMin.jar TextMin*.class
hadoop jar TextMin.jar TextMin
hadoop fs -cat features/part-r-00000
*/

public class TextMin
{
	public static enum Global_Counters 
	{
		TOTAL_TWEET_NUM, 
		TOTAL_FEATURE_NUM
	}


	/* input:  <byte_offset, tweet>
     * output: <tweet_number, tweet>
     */
	public static class Map_PreProcess extends Mapper<Object, Text, Text, Text> 
	{
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

            String tweet_number = columns[0];
            String tweet_sentiment = columns[1];
            String tweet_text = columns[3];

            // clean the text of the tweet from links..
            tweet_text = tweet_text.replaceAll("(http|https)\\:\\/\\/[a-zA-Z0-9\\-\\.]+\\.[a-zA-Z]{2,3}(\\/\\S*)?", "")
                                .replaceAll("#|@.*?\\s", "")	// mentions and hashtags...
                                .replaceAll("\\d+", "")			// numbers...
                                .replaceAll("[^a-zA-Z ]", "")	// punctuation...
                                .toLowerCase()  				// turn every character left to lowercase...
                                .trim()         				// trim the spaces before & after the whole string...
                                .replaceAll("\\s+", " "); 		// and get rid of double spaces

            if(tweet_text != null && !tweet_text.trim().isEmpty())
         		context.write(new Text(tweet_number), new Text(tweet_text));
		}
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
			context.getCounter(Global_Counters.TOTAL_TWEET_NUM).increment(1);

			String line = value.toString();

            String[] columns = line.toString().split("\t");
            String[] tweet_text = columns[1].toString().split(" ");

            for(int i=0; i<tweet_text.length; i++)
            {
                word_tweet_key.set(tweet_text[i] + "@" + columns[0]);

                context.write(word_tweet_key, one);
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
     * output: <(word@tweet), (word_count/tweet_size)>
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

			for( Text value : values)
			{
				String[] splitted_key = value.toString().split("=");

				word_list.add(splitted_key[0]);
				count_list.add(splitted_key[1]);

				total += Integer.parseInt(splitted_key[1]);
			} 
			
			// create the count/total in the list
			for(String count : count_list)
				word_by_count_list.add(count+"/"+total);
		
			// create the iterator for word_list and word_by_count_list and write the key-value pair in the context
			Iterator<String> wl = word_list.iterator();
			Iterator<String> wbcl = word_by_count_list.iterator();
			
			while (wl.hasNext() && wbcl.hasNext()) 
			{				
				context.write(new Text(wl.next() + "@" + tweet), new Text(wbcl.next().toString()));
			}
		}
    }


    /* input:  <(word@tweet), (word_count/tweet_size)>
     * output: <word, (tweet=word_count/tweet_size)>
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

    /* input:  <word, (tweet=word_count/tweet_size)>
     * output: <tweet@word, TFIDF>
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

			// access the Arraylist created above and arrange them in the required format
			for (String value : value_list)
			{
				String[] value_arr = value.split("=");
				String[] divide_data = value_arr[1].toString().split("/");
				
				tfidf = (Double.parseDouble(divide_data[0])/Double.parseDouble(divide_data[1])) * Math.log(num_of_tweets/num_of_tweets_with_this_word);
				
				context.write(new Text(value_arr[0] + "@" + key.toString()), new Text(tfidf.toString()));

				context.getCounter(Global_Counters.TOTAL_FEATURE_NUM).increment(1);
			}			
		}
    }


    /* input:  <tweet@word, TFIDF>
     * output: <NULL, tweet@word_TFIDF>
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
		//private TreeMap<Double, Text> top_features = new TreeMap<Double, Text>();
		
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



	public static void main(String[] args) throws Exception 
	{
		Path input_dir = new Path("input");
		Path processed_dir = new Path("processed_tweets");
		Path wordcount_dir = new Path("wordcount");
		Path tf_dir = new Path("tf");
    	Path tfidf_dir = new Path("tfidf");
    	Path features_dir = new Path("features");

	    Configuration conf = new Configuration();

	    FileSystem fs = FileSystem.get(conf);
	    if(fs.exists(processed_dir))
    		fs.delete(processed_dir, true);
    	if(fs.exists(wordcount_dir))
    		fs.delete(wordcount_dir, true);
    	if(fs.exists(tf_dir))
    		fs.delete(tf_dir, true);
    	if(fs.exists(tfidf_dir))
    		fs.delete(tfidf_dir, true);
    	if(fs.exists(features_dir))
    		fs.delete(features_dir, true);


    	Job preprocess_job = Job.getInstance(conf, "PreProcess Tweets");
	    preprocess_job.setJarByClass(TextMin.class);
	    preprocess_job.setMapperClass(Map_PreProcess.class);
	    //preprocess_job.setCombinerClass(Reduce_PP.class);
	    //preprocess_job.setReducerClass(Reduce_PP.class);
	    preprocess_job.setMapOutputKeyClass(Text.class);
		preprocess_job.setMapOutputValueClass(Text.class);
	    preprocess_job.setOutputKeyClass(Text.class);
	    preprocess_job.setOutputValueClass(Text.class);
	    FileInputFormat.addInputPath(preprocess_job, input_dir);
	    FileOutputFormat.setOutputPath(preprocess_job, processed_dir);
	    preprocess_job.waitForCompletion(true);


	    Job wordcount_job = Job.getInstance(conf, "Word Count");
	    wordcount_job.setJarByClass(TextMin.class);
	    wordcount_job.setMapperClass(Map_WordCount.class);
	    wordcount_job.setCombinerClass(Reduce_WordCount.class);
	    wordcount_job.setReducerClass(Reduce_WordCount.class);
	    wordcount_job.setMapOutputKeyClass(Text.class);
		wordcount_job.setMapOutputValueClass(IntWritable.class);
	    wordcount_job.setOutputKeyClass(Text.class);
	    wordcount_job.setOutputValueClass(IntWritable.class);
	    FileInputFormat.addInputPath(wordcount_job, processed_dir);
	    FileOutputFormat.setOutputPath(wordcount_job, wordcount_dir);
	    wordcount_job.waitForCompletion(true);


        int num_of_tweets = Math.toIntExact(wordcount_job.getCounters().findCounter(Global_Counters.TOTAL_TWEET_NUM).getValue());
        conf.set("num_of_tweets", String.valueOf(num_of_tweets));
        System.out.println("TOTAL TWEET NUMBER: " + num_of_tweets);


	    Job tf_job = Job.getInstance(conf, "TF");
		tf_job.setJarByClass(TextMin.class);
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
		tfidf_job.setJarByClass(TextMin.class);
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
        int num_of_features = Math.toIntExact(tfidf_job.getCounters().findCounter(Global_Counters.TOTAL_FEATURE_NUM).getValue());
        System.out.println("TOTAL NUMBER OF FEATURES: " + num_of_features);
        int features_to_keep = (num_of_features * 20) / 100;
        conf.set("features_to_keep", String.valueOf(features_to_keep));
        System.out.println("FEATURES TO KEEP: " + features_to_keep);


        Job feature_selection_job = Job.getInstance(conf, "Feature Selection");
		feature_selection_job.setJarByClass(TextMin.class);
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
  	}
}