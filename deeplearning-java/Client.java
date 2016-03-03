package socketCnn;
import java.io.*;
import java.net.*;

public class Client {

	   public static void main(String args[]) throws Exception {
	      //为了简单起见，所有的异常都直接往外抛
	     String host = "127.0.0.1";  //要连接的服务端IP地址
	     int port = 8899;   //要连接的服务端对应的监听端口
	     //与服务端建立连接
	     Socket client = new Socket(host, port);
	      //建立连接后就可以往服务端写数据了
	     Writer writer = new OutputStreamWriter(client.getOutputStream());
	      writer.write("Hello Server.");
	      writer.write("eof\n");
	      writer.flush();
	      //写完以后进行读操作
	     BufferedReader br = new BufferedReader(new InputStreamReader(client.getInputStream()));
	      StringBuffer sb = new StringBuffer();
	      String temp;
	      int index;
	      while ((temp=br.readLine()) != null) {
	         if ((index = temp.indexOf("eof")) != -1) {
	            sb.append(temp.substring(0, index));
	            break;
	         }
	         sb.append(temp);
	      }
	      System.out.println("from server: " + sb);
	      writer.close();
	      br.close();
	      client.close();
	   }
	}


