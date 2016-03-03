package socketCnn;
import java.io.*;
import java.net.*;

public class Client {

	   public static void main(String args[]) throws Exception {
	     String host = "127.0.0.1";
	     int port = 8899;
	     Socket client = new Socket(host, port);
	     Writer writer = new OutputStreamWriter(client.getOutputStream());
	      writer.write("Hello Server.");
	      writer.write("eof\n");
	      writer.flush();
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


