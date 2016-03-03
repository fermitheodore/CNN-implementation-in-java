/**
*@bief the CNN class used to estiblish the CNN networks.
*
*@author Tianyi Liu
*@date 2015-5-23
*@version 0.1
*
*@history
*     <author>      <date>     <version>     <description>
*    Tianyi Liu   2015-05-23      0.1        the cnn class
*    Tianyi Liu   2015-05-24      0.1          bp
*    Tianyi Liu   2015-05-29      0.1          bp修改	
*    Tianyi Liu   2015-05-29      0.1         bp算法批处理
*    Tianyi Liu   2015-05-31      0.2         理论分析并行效率
*
*
*/
import java.awt.Image;
import java.awt.image.ColorModel;
import java.awt.image.MemoryImageSource;
import java.awt.image.PixelGrabber;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import java.util.*;

public class Cnn{
	public int n_layers;//num of layers
	
	public double[][]target_out_table;	
	public double[] pic_error_vector;

	public input_layer[] input_layers;
	public conv_layer Conv1,Conv2;
	public samp_layer Samp1,Samp2;
	public hidden_layer hidden;
	public output_layer output;	

	public static int num_person;
	public int num_picture;
	
  	Cnn(int num,File[] list){//构造函数实现Cnn网络到初始化
		n_layers = 7;
		this.num_person = num;
		int fileCount = 0;
		//统计输入照片数量
		for (File folder:list){
			File[] filelist = folder.listFiles();
			for(File file:filelist){
				if(file.isFile()){
					fileCount++;
				}
			}
		}
		this.num_picture = fileCount;
		//建立对象数组
	 	input_layers = new input_layer[fileCount];
		target_out_table = new double[fileCount][num_person];
		pic_error_vector = new double[num_person];
		for(int i=0;i<fileCount;i++){
			input_layer g = new input_layer();
			input_layers[i] = g;
		}
		//从磁盘读入图像，放入对象数组
		fileCount = 0;
		int folderCount =0;
		for (File folder1:list){
			//获得人的名字
			String person_name = folder1.getName();
			File[] filelist1 = folder1.listFiles();
			for(File file:filelist1){
				if(file.isFile()){
					target_out_table[fileCount][folderCount] = 1;
					input_layers[fileCount++].readImage(file,person_name);
				}
				
			}
			folderCount++;
		}
		for(int i=0;i<num_picture;i++){
			for(int j=0;j<num_person;j++){
				if(target_out_table[i][j]==0)
					target_out_table[i][j] = -1;
			}
		}
		System.out.println(fileCount);
		//fileCount中已经存储了照片个数
		//初始化网络参数
		cnn_init(input_layers[0].w,input_layers[0].h);
    	}

	/*public void cnn_test(String pic_path){
		File folder2 = new File(pic_path);
		File[] list = folder.listFiles();
		int fileCount = 0;
		for(File file:list){
			if(file.isFile()){
				fileCount++;
			}
		}
	}*/
   	public void cnn_train(){
		int train_num = 0;
		double pictures_total_error = 0;
		double avg_single_error = 0;
		System.out.println("开始训练");
		//long t1,t1_start,t1_end,t2_start,t2_end,t2;
		long startTime = System.currentTimeMillis();
		do{
			//System.out.println();
			//t1=t2=0;
			pictures_total_error = 0;
			for(int i=0;i<num_picture;i++){
				//t1_start = System.currentTimeMillis();
				pictures_total_error += cnn_ff(i);//完成前向传递
				//t1_end = System.currentTimeMillis();
				//t1 +=(t1_end-t1_start);
				//System.out.println("t1:"+t1);
				
				//t2_start = System.currentTimeMillis();
				cnn_bp();
				//t2_end = System.currentTimeMillis();
				//t2 +=(t2_end-t2_start);

				//System.out.println("t2:"+(t2_end-t2_start));
    			}
			//System.out.println("avg_t1:"+t1/num_picture);			
			//System.out.println("avg_t2:"+t2/num_picture);			

			//long t3_start = System.currentTimeMillis();
			//cnn_ajust_parameter();
			//cnn_send_parameter();
			//long t3_end = System.currentTimeMillis();
			//System.out.println("t3:"+(t3_end-t3_start));

			avg_single_error = pictures_total_error/num_picture;

		System.out.println("平均误差："+avg_single_error);
		}while(!(avg_single_error<=4.0||++train_num>30-1));//当训练次数超过1000或者误差很小了，就停止训练
		//}while(!(++train_num>30-1));
		long endTime = System.currentTimeMillis();
		System.out.println("训练耗时:"+(endTime-startTime)+"ms");
	}
	public void cnn_send_parameter(){
		//发送4个层的各个权值和偏执
		






	}
	public void cnn_ajust_parameter(){
		output.ajust_parameter();
		hidden.ajust_parameter();
		Conv1.ajust_parameter();
		Samp1.ajust_parameter();
		Conv2.ajust_parameter();
		Samp2.ajust_parameter();
	}
	private double cnn_ff(int i){
		double total_error=0;
		Conv1.convolutional(input_layers[i].inputPixArray);
		Samp1.sampling(Conv1.conv_value);
		Conv2.convolutional(Samp1.samp_value);
		Samp2.sampling(Conv2.conv_value);

		hidden.hidden_calculate(Samp2.samp_value);
		output.output_calculate(hidden.hidden_value);
		double tempvalue = 0;
		for(int j=0;j<num_person;j++){
			tempvalue = output.output_value[j]-target_out_table[i][j];
			pic_error_vector[j] = tempvalue;
			total_error += tempvalue*tempvalue;
		}
		//System.out.println("每张误差："+total_error);
		return total_error;
	}

	private void cnn_bp(){
		//System.out.println("进入bp");
		output.bp(pic_error_vector);
		hidden.bp(output.last_d_out);
		Samp2.bp(hidden.last_d_out);
		Conv2.bp(Samp2.last_d_out);
		Samp1.bp(Conv2.last_d_out);
		Conv1.bp(Samp1.last_d_out);
	}
   	private void cnn_init(int pic_w,int pic_h){
		
		Conv1  = new conv_layer(6, pic_w, pic_h,5,5,6);//特征图个数，输入图片宽，高，卷积核行，列，卷积核个数
		Samp1  = new samp_layer(Conv1.num_figuremap,Conv1.figuremap_h,Conv1.figuremap_w);//卷积层特征图个数，卷积图的行，列
		Conv2  = new conv_layer(16,Samp1.samp_c,Samp1.samp_r,5,5,16*6);//16个figuremap,均用6个卷积核。
		//特征图个数，输入矩阵的列，行，卷积核个数，卷积核行，列
		Samp2  = new samp_layer(Conv2.num_figuremap,Conv2.figuremap_h,Conv2.figuremap_w);
		
		output = new output_layer(num_person);//传入输出层节点个数
		hidden = new hidden_layer(10*num_person,Samp2.num_sampmap,Samp2.samp_r,Samp2.samp_c);//传入隐藏层节点个数
		System.out.println("cnn_init"+num_person);
   	}
}
/**********************************父层及接口**************************************/

class layers{
	public int n_node;
	public double layer_bias;
	
	public static double sigmoid(double x){
		return 1.0 / (1.0 + Math.pow(Math.E,-x));
	}
	public static double arcsigmod(double x){
		return x*(1-x);
	}
	public static double tanh(double x){
		return 1-2/(1 + Math.pow(Math.E,2*x));
	}
	public static double arctanh(double x){
		//return 4.0/(2+Math.pow(Math.E,2*x)*Math.pow(Math.E,-2*x));
		return (1+x)*(1-x);
	}
}

interface calculation{
	public static double learn_ratio = 0.6;
	
	//public double calculate();
}
/***************输入层*********************/
final class input_layer{

	public String person_name;
	public double currentPixArray[][];
	public double inputPixArray[][];
	public int w,h;
	private BufferedImage bi;
	//读取图片,放入currentPixArray[][]数组中
	public void readImage(File file, String _name){
		person_name = _name;
		try{
			bi = ImageIO.read(file);
			currentPixArray = Image2Matrix(bi);
			inputPixArray = new double[h][w];
			double maxvalue=0;
			double minvalue=300;
			for(int i = 0;i<h;i++){
				for(int j = 0;j<w;j++){
					maxvalue = currentPixArray[i][j]>maxvalue ? currentPixArray[i][j]:maxvalue;
					minvalue = currentPixArray[i][j]<minvalue ? currentPixArray[i][j]:minvalue;
				}
			}
			for(int i = 0;i<h;i++){
				for(int j = 0;j<w;j++){
					//System.out.println(currentPixArray[i][j]);
					inputPixArray[i][j] = 2* (currentPixArray[i][j]-minvalue)/(maxvalue-minvalue)-1;
					//System.out.println(inputPixArray[i][j]);
				}
			}
		} catch (IOException ex){
			System.out.println(ex);
		}
	}
	//将图片转化为2维矩阵
	private double[][] Image2Matrix(BufferedImage im){
		w = im.getWidth();
		h = im.getHeight();
		int minx = im.getMinX();
		int miny = im.getMinY();
		int[] rgb = new int[3];
		double array[][] = new double[h][w];
		try{
			for(int i = miny;i<h;i++){
				for(int j = minx;j<w;j++){
					int pixel = im.getRGB(i,j);
					array[i][j] = (double)(pixel & 0xff);
				}
			}
		}catch(Exception ex){
			ex.printStackTrace();
		}
		return array;
	}
}
/****************卷积层*******************/
final class conv_layer extends layers implements calculation{
	public int num_total_node;//该层所有神经元个数
	public int num_figuremap;
	public int num_figuremap_node;
	public int figuremap_w;
	public int figuremap_h;

	public int core_width,core_hight,number_core,ratio_core_map;
	//三维的原因是因为，第一维为哪个卷积核，后两维为卷积核的行列索引。
	public double[][][] conv_core;
	public double[][][] delta_conv_core;

	public double[] conv_bias;//每一个卷积核有一个偏置
	public double[] delta_conv_bias;

	//第一维为哪个figuremap，第二维为figuremap内的索引。
	public double[][][] conv_value;


	public double[][][] input_value;
	private double[][][] upper_out;//上层的输出,哪个特征图的哪行哪列
	public double[][][] last_d_out;//上层的输出误差;第m个特征图，第x行，第y列
	private double[][][] d_in;//输入误差

	conv_layer(int num_map, int pic_w, int pic_h, int core_w, int core_h, int num_core){
		this.number_core = num_core;
		this.core_width = core_w;
		this.core_hight = core_h;

		this.num_figuremap = num_map;
		this.ratio_core_map = num_core/num_map;
		this.figuremap_w = pic_w - core_w + 1;
		this.figuremap_h = pic_h - core_h + 1;
		this.num_figuremap_node = figuremap_w * figuremap_h;
		this.num_total_node = num_figuremap * num_figuremap_node;
		
		conv_bias = new double[num_figuremap];
		delta_conv_bias = new double[num_figuremap];

		conv_core = new double[num_core][core_h][core_w];
		delta_conv_core=new double[num_core][core_h][core_w];

		upper_out = new double[num_map][pic_h][pic_w];
		//last_d_out= new double[][][];
		Random a = new Random();
		for(int i=0;i<num_figuremap;i++){
			conv_bias[i] = a.nextDouble()*0.02-0.01;
		}
		for(int i=0;i<num_core;i++)
			for(int j=0;j<core_h;j++)
				for(int k=0;k<core_w;k++)
					conv_core[i][j][k] = a.nextDouble()*0.02-0.01;
	}	
	public void bp(double array[][][]){
		d_in = new double[num_figuremap][figuremap_h][figuremap_w];
		for(int k=0;k<num_figuremap;k++){
			for(int i=0;i<figuremap_h;i++){
				for(int j=0;j<figuremap_w;j++){
					d_in[k][i][j]=array[k][i][j]*arctanh(conv_value[k][i][j]);//计算出卷积层输入误差
					delta_conv_bias[k]-=learn_ratio*d_in[k][i][j];//偏置更新,负梯度
				}
			}
		}
		//计算前层输出误差(从卷积结果往前投射，叠加到last_d_out矩阵)
		last_d_out = new double[ratio_core_map][figuremap_h+core_hight-1][figuremap_w+core_width-1];
		for(int k=0;k<num_figuremap;k++){
			for(int m=0;m<ratio_core_map;m++){
				for(int x=0;x<figuremap_h;x++){
				for(int y=0;y<figuremap_w;y++){
					for(int r=0;r<core_hight;r++){
					for(int c=0;c<core_width;c++){
						last_d_out[m][x+r][y+c]+=d_in[k][x][y]*conv_core[k*ratio_core_map+m][r][c];
					}}
			}}
		}}

		//计算权值更新
		for(int k=0;k<num_figuremap;k++){
		for(int m=0;m<ratio_core_map;m++){
			for(int r=0;r<core_hight;r++){
			for(int c=0;c<core_width;c++){
				for(int x=0;x<figuremap_h;x++){
				for(int y=0;y<figuremap_w;y++){
					delta_conv_core[k*ratio_core_map+m][r][c] -= learn_ratio*d_in[k][x][y]*upper_out[k][x+r][y+c];//负梯度
				}}
			}}
		}}

	}
	public void ajust_parameter(){
		for(int k=0;k<num_figuremap;k++){
			conv_bias[k]+= delta_conv_bias[k]/Cnn.num_person;
			for(int i=0;i<core_hight;i++)
				for(int j=0;j<core_width;j++)
					conv_core[k][i][j]+=delta_conv_core[k][i][j]/Cnn.num_person;
		}
	}
	public void convolutional(double array[][]){
		conv_value = new double[num_figuremap][figuremap_h][figuremap_w];
		input_value= new double[num_figuremap][figuremap_h][figuremap_w];
		if(num_figuremap == number_core){
			for(int num=0;num<num_figuremap;num++){
			for(int i=0;i<figuremap_h;i++)	{
				for(int j=0;j<figuremap_w;j++){
					for(int k=0;k<core_hight;k++){
					for(int l=0;l<core_width;l++){
						conv_value[num][i][j]+=conv_core[num][k][l]*array[k+i][l+j];
					}}
					input_value[num][i][j] = conv_value[num][i][j]+conv_bias[num];
					conv_value[num][i][j]=tanh(input_value[num][i][j]);
				}
			}}
		}}
	public void convolutional(double array[][][]){//采用全连接方式，卷积核个数一定是figuremap个数的6倍
			conv_value = new double[num_figuremap][figuremap_h][figuremap_w];
			input_value= new double[num_figuremap][figuremap_h][figuremap_w];
			int index = 0;
			for(int num=0;num<num_figuremap;num++){//map
				index =0;
				for(int i=0;i<figuremap_h;i++){
				for(int j=0;j<figuremap_w;j++){
					for(int num_ratio=0+6*index;num_ratio<ratio_core_map+6*index;num_ratio++){//遍历6个卷积核
						for(int k=0;k<core_hight;k++){
						for(int l=0;l<core_width;l++){
							input_value[num][i][j]+=conv_core[num_ratio][k][l]*array[index][k+i][l+j];
						}}
					}
				}}
				index++;
				//当所有6个卷积完，即将开始下6个卷积之前，进行值调整。
				for(int i=0;i<figuremap_h;i++){
					for(int j=0;j<figuremap_w;j++){
						input_value[num][i][j]=input_value[num][i][j]/6.0+conv_bias[num];
						conv_value[num][i][j] = tanh(input_value[num][i][j]);
						//System.out.println(input_value[num][i][j]);
					}
				}
			}
		
		}
}
/*****************子采样层****************/
final class samp_layer extends layers implements calculation{
	public int num_total_node;//该层所有神经元个数
	public int num_sampmap;//采样层figuremap个数
	public int num_sampmap_node;//一个figuremap有多少个节点
	public int samp_r,samp_c;
	//卷积层有多少个figuremap,采样层就有多少个weight。多少个偏置。
	public double[] samp_weight;
	public double[] delta_samp_weight;

	public double[] samp_bias;
	public double[] delta_samp_bias;

	//第一维是哪个figuremap,后两维为map内索引。
	public double[][][] samp_value;
	
	public double[][][] input_value;
	private double[][][] upper_out;//上层的输出,哪个特征图的哪行哪列
	public double[][][] last_d_out;//上层的输出误差;第m个特征图，第x行，第y列
	private double[][][] d_in;//输入误差

	samp_layer(int num_convmap, int convmap_r, int convmap_c){
		samp_r = convmap_r/2;
		samp_c = convmap_c/2;

		samp_bias = new double[num_convmap];
		delta_samp_bias=new double[num_convmap];
		samp_weight = new double[num_convmap];
		delta_samp_weight=new double[num_convmap];

		num_sampmap = num_convmap;
		num_sampmap_node = samp_r*samp_c;
		num_total_node = num_sampmap_node * num_sampmap;
		
		last_d_out = new double[num_convmap][convmap_r][convmap_c];
		Random a = new Random();
		for (int i=0;i<num_sampmap;i++){
			samp_weight[i] = a.nextDouble()*0.02-0.01;
			samp_bias[i]   = a.nextDouble()*0.02-0.01;
		}

	}
	public void bp(double array[][][]){
		d_in = new double[num_sampmap][samp_r][samp_c];
		double total_d_in;//计算偏置改变，更新偏置
		for(int k=0;k<num_sampmap;k++){
			total_d_in =0;
			for(int i=0;i<samp_r;i++){
				for(int j=0;j<samp_c;j++){
					d_in[k][i][j]=array[k][i][j]*arctanh(samp_value[k][i][j]);//计算输入误差
					total_d_in-=learn_ratio*d_in[k][i][j];//负梯度
				}
			}
			delta_samp_bias[k]-=learn_ratio*total_d_in;//计算偏置改变量，更新偏置。
			//System.out.println("samp_bias:"+samp_bias[k]);
		}
		//System.out.println(num_sampmap);
		//计算前一层的输出误差
		for(int k=0;k<num_sampmap;k++){
			for(int i=0;i<samp_r*2;i++){
				for(int j=0;j<samp_c*2;j++){
					last_d_out[k][i][j]=samp_weight[k]*d_in[k][i/2][j/2];//默认采用2*2的下采样
					//System.out.println("samp_last_d_out:"+last_d_out[k][i][j]);
				}
			}
		}

		//计算权值改变量，更新权值
		for(int k=0;k<num_sampmap;k++){
			for(int i=0;i<samp_r*2;i++){
				for(int j=0;j<samp_c;j++){
					delta_samp_weight[k]-=learn_ratio*d_in[k][i/2][j/2]*upper_out[k][i][j];//负梯度
				}
			}
		}
	}
	public void ajust_parameter(){
		for(int k=0;k<num_sampmap;k++){
			samp_bias[k]+=delta_samp_bias[k]/Cnn.num_person;
			samp_weight[k]+=delta_samp_weight[k]/Cnn.num_person;
		}
	}
	public void sampling(double array[][][]){
		samp_value = new double[num_sampmap][samp_r][samp_c];
		input_value= new double[num_sampmap][samp_r][samp_c];
		upper_out  = new double[num_sampmap][samp_r*2][samp_c*2];
		upper_out  = array;
		for(int num=0;num<num_sampmap;num++){
			for(int i=0;i<samp_r;i++){
			for(int j=0;j<samp_c;j++){
			    samp_value[num][i][j]=array[num][2*i][2*j]+array[num][2*i][2*j+1]+array[num][2*i+1][2*j]+array[num][2*i+1][2*j+1];
			    input_value[num][i][j]=samp_value[num][i][j]*samp_weight[num]+samp_bias[num];
			    samp_value[num][i][j]=tanh(input_value[num][i][j]);
			}}
		}
	}
}
/******************隐藏层*****************/
final class hidden_layer extends layers implements calculation{
	public int num_node;
	//由于该层为全连接层，所以，第一维为本层第几个神经元
	//第二维为前一采样层的num_total_node
	public double[][][][] hidden_weight;
	public double[][][][] delta_hidden_weight;

	public double[] hidden_bias;
	public double[] delta_hidden_bias;

	//隐藏层值数组长度为该层num_total_node的值
	public double[] hidden_value;
	private int last_figuremap_num,last_figuremap_r,last_figuremap_c;

	public double[] input_value;
	public double[][][] upper_out;//上层的输出,哪个特征图的哪行哪列
	public double[][][] last_d_out;//上层的输出误差;第m个特征图，第x行，第y列
	public double[] d_in;//输入误差
	hidden_layer(int num, int map_num, int map_r, int map_c){
		num_node = num;
		last_figuremap_num = map_num;
		last_figuremap_r   = map_r;
		last_figuremap_c   = map_c;

		hidden_bias = new double[num];
		delta_hidden_bias =new double[num];
		hidden_weight = new double[num][map_num][map_r][map_c];
		delta_hidden_weight=new double[num][map_num][map_r][map_c];

		Random a = new Random();

		for(int i=0;i<num;i++){
		for(int j=0;j<map_num;j++){
			for(int m=0;m<map_r;m++){
			for(int n=0;n<map_c;n++){
				hidden_weight[i][j][m][n]=a.nextDouble()*0.02-0.01;
			}}
		}}
	}
	public void ajust_parameter(){
		for(int k=0;k<num_node;k++){
			hidden_bias[k]+=delta_hidden_bias[k]/Cnn.num_person;
			for(int m=0;m<last_figuremap_num;m++)
				for(int i=0;i<last_figuremap_r;i++)
					for(int j=0;j<last_figuremap_c;j++)
						hidden_weight[k][m][i][j]+=delta_hidden_weight[k][m][i][j]/Cnn.num_person;
		}
	}
	public void bp(double array[]){
		d_in = new double[num_node];
		for(int i=0;i<array.length;i++){
			d_in[i]=array[i]*arctanh(hidden_value[i]);//求出输入误差，也即偏置的改变量。
			delta_hidden_bias[i]-=learn_ratio*d_in[i];//隐藏层偏置更新，负梯度
		}
		
		//计算前一层的输出误差
		for(int k=0;k<last_figuremap_num;k++){
			for(int i=0;i<last_figuremap_r;i++){
			for(int j=0;j<last_figuremap_c;j++){
				for(int m=0;m<num_node;m++)
					last_d_out[k][i][j]+=d_in[m]*hidden_weight[m][k][i][j];
			}
			}
		}

		//隐藏层权值更新
		for(int i=0;i<num_node;i++){
		for(int j=0;j<last_figuremap_num;j++){
			for(int m=0;m<last_figuremap_r;m++){
			for(int n=0;n<last_figuremap_c;n++){
				//hidden_weight[i][j][m][n]+=upper_out[j][m][n]*d_in[j*last_figuremap_r*last_figuremap_c+m*last_figuremap_c+n];
				delta_hidden_weight[i][j][m][n]-=learn_ratio*upper_out[j][m][n]*d_in[i];//负梯度
			}
			}
		}
		}

	}
	public void hidden_calculate(double array[][][]){
		double tempvalue=0;
		hidden_value = new double[num_node];
		input_value  = new double[num_node];
		upper_out    = new double[last_figuremap_num][last_figuremap_r][last_figuremap_c];
		last_d_out    = new double[last_figuremap_num][last_figuremap_r][last_figuremap_c];
		upper_out = array;
		for(int i=0;i<num_node;i++){
			for(int j=0;j<last_figuremap_num;j++){
			for(int m=0;m<last_figuremap_r;m++){
			for(int n=0;n<last_figuremap_c;n++){
				tempvalue+=hidden_weight[i][j][m][n]*array[j][m][n];
			}}}
		tempvalue+=hidden_bias[i];
		input_value[i] = tempvalue;
		hidden_value[i] = tanh(tempvalue);
		}
	}
}
/*****************输出层******************/
final class output_layer extends layers implements calculation{
	public int num_node;
	//第一个标代表输出层哪个神经元，第二个标表示该神经元权重
	public double[][] layer_weight;
	public double[][] delta_layer_weight;//全部加和，最后求平均

	public double[] layer_bias;
	public double[] delta_layer_bias;//全部加和，最后求平均

	public double[] output_value;
	public double[] input_value;
	
	private double[] upper_out;//隐藏层的输出
	public double[] last_d_out;//上层的输出误差
	private double[] d_in;//输出层到隐含误差

	output_layer(int num){
		this.num_node = num;
		layer_bias = new double[num];
		delta_layer_bias = new double[num];
		layer_weight = new double[num][10*num];
		delta_layer_weight=new double[num][10*num];
		Random a = new Random();
		for (int i=0;i<num;i++){
			for(int j=0;j<10*num;j++){
				layer_weight[i][j] = a.nextDouble()*0.02-0.01;
			}
		}
	}
	public void ajust_parameter(){
		for(int i=0;i<num_node;i++){
			layer_bias[i]+=delta_layer_bias[i]/Cnn.num_person;	
			for(int j=0;j<num_node*10;j++){
				layer_weight[i][j]+=delta_layer_weight[i][j]/Cnn.num_person;
			}

		}
				
	}
	public void bp(double array[]){//array[]中的内容就相当于输出误差
		last_d_out = new double[10*num_node];
		d_in  = new double[num_node];
		double tempvalue=0;
		//偏置更新
		for(int k=0;k<array.length;k++){
			tempvalue=arctanh(output_value[k])*array[k];//计算输出层偏置改变量
			d_in[k]=tempvalue;
			delta_layer_bias[k] -= learn_ratio*tempvalue;//偏置更新,负梯度
		}
		//计算出上层的输出误差
		for(int i=0;i<10*num_node;i++){
			for(int j=0;j<num_node;j++){
				last_d_out[i]+=layer_weight[j][i]*d_in[j];//计算出上层的输出误差
			}
		}
		//计算出权值改变，并更新权值
		for(int k=0;k<array.length;k++){
			for(int j=0;j<10*array.length;j++){
				delta_layer_weight[k][j]-=learn_ratio*d_in[k]*upper_out[j];//负梯度
				
			}
		}
	}
	public void output_calculate(double array[]){
		output_value = new double[num_node];
		input_value  = new double[num_node];
		upper_out    = new double[num_node];
		upper_out    = array;
		for(int i=0;i<num_node;i++){
			double tempvalue = 0;
			for(int j=0;j<10*num_node;j++){
				tempvalue += layer_weight[i][j]*array[j];
			}
			tempvalue += layer_bias[i];
			input_value[i] = tempvalue;
			output_value[i] = tanh(tempvalue);
		}
	}
	
}
