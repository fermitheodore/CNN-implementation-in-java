import java.io.*;
import java.net.*;
public class CNNDEMO_client{
   public static void main(String[] args) throws Exception {//自动训练
/**********************************************
args[0]:路径
路径+s[num]+num.jpg即为待训练图片
***********************************************/
	int folderCount = 0;
/****检查训练路径***/
	if(args.length == 0){
		System.out.println("请输入训练图片路径！");
		return;
	}
/***统计文件夹个数**/
	File folder = new File(args[0]);
	File[] list = folder.listFiles();
	for(File file:list){
		if(file.isDirectory()){
			folderCount++;
		}
	}
	if (folderCount == 0){
		System.out.println("没有文件夹，即没有人");
		return;
	}
/***根据输入构造网络***/
        Cnn cnnnet = new Cnn(folderCount,list);

        cnnnet.cnn_train();
	String host = "192.168.1.100";
	int port = 8899;
	//与服务器建立连接
	Socket client = new Socket(host,port);
	//建立连接后，就可以往服务器写数据了
	Writer writer = new OutputStreamWriter(client.getOutputStream());

	writer.write(cnnnet.output.delta_layer_weight);
	writer.write(cnnnet.output.delta_layer_bias);
	writer.write(cnnnet.hidden.delta_hidden_weight);
	writer.write(cnnnet.hidden.delta_hidden_bias);

	writer.write(cnnnet.Conv1.delta_conv_core);
	writer.write(cnnnet.Conv1.delta_conv_bias);
	writer.write(cnnnet.Conv2.delta_conv_core);
	writer.write(cnnnet.Conv2.delta_conv_bias);

	writer.write(cnnnet.Samp1.delta_samp_weight);
	writer.write(cnnnet.Samp1.delta_samp_bias);
	writer.write(cnnnet.Samp2.delta_samp_weight);
	writer.write(cnnnet.Samp2.delta_samp_bias);

	writer.write("eof\n");	
	writer.flush();
	//写完以后进行读操作
	BufferedReader br = new BufferedReader(new InputStreamReader(client.getInputStream()));
	StringBuffer sb = new StringBuffer();
	String temp;
	int index;
	while((temp=br.readLine())!=null){
		if((index = temp.indexOf("eof"))!=-1){
			sb.append();
		}
			
	}	

	System.out.println(folderCount);
   }
}
