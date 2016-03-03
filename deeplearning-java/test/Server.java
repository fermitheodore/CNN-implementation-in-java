package socketCnn;

import java.io.*;
import java.net.*;

public class Server {
	 
	   public static void main(String args[]) throws IOException {
	     int port = 8899;
	     ServerSocket server = new ServerSocket(port);
	      while (true) {
	         Socket socket = server.accept();
	         new Thread(new Task(socket)).start();
	      }
	   }
	   
	   static class Task implements Runnable {
	 
	      private Socket socket;
	      
	      public Task(Socket socket) {
	         this.socket = socket;
	      }
	      
	      public void run() {
	         try {
	            handleSocket();
	         } catch (Exception e) {
	            e.printStackTrace();
	         }
	      }
	      
	      private void handleSocket() throws Exception {
	         BufferedReader br = new BufferedReader(new InputStreamReader(socket.getInputStream()));
	         StringBuilder sb = new StringBuilder();
	         String temp;
	         int index;
	         while ((temp=br.readLine()) != null) {
	            System.out.println(temp);
	            if ((index = temp.indexOf("eof")) != -1) {
	             sb.append(temp.substring(0, index));
	                break;
	            }
	            sb.append(temp);
	         }
	         System.out.println("from client: " + sb);
	         Writer writer = new OutputStreamWriter(socket.getOutputStream());
	         writer.write("Hello Client.");
	         writer.write("eof\n");
	         writer.flush();
	         writer.close();
	         br.close();
	         socket.close();
	      }
	   }
	}

