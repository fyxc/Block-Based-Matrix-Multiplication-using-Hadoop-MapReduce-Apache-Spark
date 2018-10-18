package edu.ruoxianj.hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.IOException;

public class BlockMult {
    /**
    for matrix A, convert (1,1),[(1,1,2),(1,2,1),(2,2,3)] to <(1,1) ; A,(1,1),[(1,1,2),(1,2,1),(2,2,3)]> and <(1,2) ; A,(1,1),[(1,1,2),(1,2,1),(2,2,3)]> and <(1,3) ; A,(1,1),[(1,1,2),(1,2,1),(2,2,3)]>
     */
    public static class LeftMatrixMapper extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            String line = value.toString();
            String blockIndices = line.substring(0, 5);
            String content = line.substring(6);
            String[] blockRowAndColIndex = blockIndices.substring(1, blockIndices.length() - 1).split(",");
            int blockRowIndex = Integer.parseInt(blockRowAndColIndex[0]);
            int blockColIndex = Integer.parseInt(blockRowAndColIndex[1]);
            Text outputKey = new Text();
            Text outputValue = new Text();
            for (int i = 1; i <= 3; i ++) {
                String generatedBlockRowAndColIndex = "(" + blockRowIndex + "," + i + ")";
                outputKey.set(generatedBlockRowAndColIndex);
                outputValue.set("A," + line);
                context.write(outputKey, outputValue);
            }
        }
    }

    /**
     *  for matrix B, convert (1,1),[(1,1,2),(1,2,1),(2,2,3)] to <(1,1) ; B,(1,1),[(1,1,2),(1,2,1),(2,2,3)]> and <(2,1) ; B,(1,1),[(1,1,2),(1,2,1),(2,2,3)]> and <(3,1) ; B,(1,1),[(1,1,2),(1,2,1),(2,2,3)]>
     */
    public static class RightMatrixMapper extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            String line = value.toString();
            String blockIndices = line.substring(0, 5);
            String content = line.substring(6);
            String[] blockRowAndColIndex = blockIndices.substring(1, blockIndices.length() - 1).split(",");
            int blockRowIndex = Integer.parseInt(blockRowAndColIndex[0]);
            int blockColIndex = Integer.parseInt(blockRowAndColIndex[1]);
            Text outputKey = new Text();
            Text outputValue = new Text();
            for (int i = 1; i <= 3; i ++) {
                String generatedBlockRowAndColIndex = "(" + i + "," + blockColIndex + ")";
                outputKey.set(generatedBlockRowAndColIndex);
                outputValue.set("B," + line);
                context.write(outputKey, outputValue);
            }
        }
    }

    /**
     *  compute C^i,j = A^i,k * B^k,j for all 1 <= k <= 3
     */
    public static class MatrixReducer extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Text outputKey = new Text();
            Text outputValue = new Text();
            int[][][][] matrices = new int[2][3][2][2];
            int[][] sum = new int[2][2];
            int[][] tmp = null;
            for (Text value : values) {
                String line = value.toString();
                String lable = line.substring(0, 1);
                String indices = line.substring(2, 7);
                String content = line.substring(8);
                String[] blockRowAndColIndex = indices.substring(1, indices.length() - 1).split(",");
                int blockRowIndex = Integer.parseInt(blockRowAndColIndex[0]);
                int blockColIndex = Integer.parseInt(blockRowAndColIndex[1]);
                int dimOne = 0;
                int dimTwo = 0;
                if (lable.equals("A")) {
                    dimOne = 0;
                    dimTwo = blockColIndex - 1;
                } else {
                    dimOne = 1;
                    dimTwo = blockRowIndex - 1;
                }
                String[] elems = content.substring(1, content.length() - 1).split("\\),");
                int len = elems.length;
                for (int i = 0; i < len; i ++) {
                    String elem = elems[i];
                    if (i != len - 1) {
                        elem += ")";
                    }
                    String[] indexAndValue = elem.substring(1, elem.length() - 1).split(",");
                    int rowIndex = Integer.parseInt(indexAndValue[0]) - 1;
                    int colIndex = Integer.parseInt(indexAndValue[1]) - 1;
                    int elemValue = Integer.parseInt(indexAndValue[2]);
                    matrices[dimOne][dimTwo][rowIndex][colIndex] = elemValue;
                }
            }
            for (int i = 0; i < 3; i ++) {
                tmp = new int[2][2];
                for (int j = 0; j < 2; j ++) {
                    for (int k = 0; k < 2; k ++) {
                        int blockValue = 0;
                        for (int z = 0; z < 2; z ++) {
                            int firstElem = matrices[0][i][j][z];
                            int secondElem = matrices[1][i][z][k];
                            blockValue += firstElem * secondElem;
                        }
                        tmp[j][k] = blockValue;
                    }
                }
                for (int j = 0; j < 2; j ++) {
                    for (int k = 0; k < 2; k ++) {
                        sum[j][k] += tmp[j][k];
                    }
                }
            }
            StringBuilder sb = new StringBuilder();
            sb.append(key.toString());
            sb.append(",");
            sb.append("[");
            for (int j = 0; j < 2; j ++) {
                for (int k = 0; k < 2; k ++) {
                    if (sum[j][k] != 0) {
                        sb.append("[");
                        sb.append(j + 1);
                        sb.append(",");
                        sb.append(k + 1);
                        sb.append(",");
                        sb.append(sum[j][k]);
                        sb.append("]");
                        sb.append(",");
                    }
                }
            }
            sb.deleteCharAt(sb.length() - 1);
            sb.append("]");
            outputValue.set(sb.toString());
            context.write(null, outputValue);
        }
    }

    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();

        Job job = new Job(conf, "BlockMult");

        job.setJarByClass(BlockMult.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setReducerClass(MatrixReducer.class);

        job.setOutputFormatClass(TextOutputFormat.class);

        MultipleInputs.addInputPath(job, new Path(args[0]), TextInputFormat.class, LeftMatrixMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, RightMatrixMapper.class);
        FileOutputFormat.setOutputPath(job, new Path(args[2]));

        job.waitForCompletion(true);
    }
}
