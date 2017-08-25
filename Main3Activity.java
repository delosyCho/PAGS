package com.example.administrator.opencvtest;

import android.content.Intent;
import android.os.Bundle;
import android.app.Activity;
import android.os.Handler;
import android.os.Message;
import android.view.View;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketAddress;
import java.net.UnknownHostException;

public class Main3Activity extends Activity {

    public Activity myActivity = this;

    //  TCP연결 관련
    private Socket clientSocket;
    private BufferedReader socketIn;
    private PrintWriter socketOut;
    private int port = 11551;
    private final String ip = "192.168.0.8";
    private MyHandler1 myHandler;
    private MyThread1 myThread;

    public TextView tv;
    public EditText nickText;
    public EditText msgText;
    public ScrollView sv;

    //private Handler mHandler = null;
    String response = "";
    public String nickName;

    Intent intent = null;
    String Question = null;

    boolean is_Registered_as_MC = false;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main3);

        sv = (ScrollView)findViewById(R.id.scrollView1);
        tv = (TextView)findViewById(R.id.text01);
        nickText = (EditText)findViewById(R.id.connText);
        msgText = (EditText)findViewById(R.id.chatText);

        intent = getIntent();
        Question = intent.getStringExtra("question");

        logger("채팅을 시작합니다.");

        boolean notRegisteredYet = true;

        new Thread() {
            public void run() {
                //connect(server, port , nickName);

                try {
                    clientSocket = new Socket(ip, port);
                    socketIn = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
                    socketOut = new PrintWriter(clientSocket.getOutputStream(), true);
                } catch (Exception e) {
                    logger(e.toString());
                }
            }
        }.start();

        myHandler = new MyHandler1();
        myThread = new MyThread1();
        myThread.start();

        logger("Coneeted!");

    }

    private Runnable showUpdate = new Runnable() {

        public void run() {
            Toast.makeText(Main3Activity.this, "Coming word: " + " ", Toast.LENGTH_SHORT).show();
        }

    };

    private Thread checkUpdate = null;

    public void connBtnClick(View v) {
        switch (v.getId()) {
            case R.id.connBtn: // 접속버튼
                try{
                    if(is_Registered_as_MC == false){
                        new Thread() {
                            public void run() {
                                //connect(server, port , nickName);
                                try{
                                    nickName = nickText.getText().toString();

                                    socketOut.println("#0@" + nickName + " : " + "@None");

                                    is_Registered_as_MC = true;
                                }catch (Exception e){
                                    logger(e.toString());
                                    logger("Register Error!");
                                }
                            }
                        }.start();

                    }
                }catch (Exception ex){

                }

                break;
            case R.id.sendBtn: // 메세지 보내기 버튼
                if (socketIn != null) {

                    new Thread() {
                        public void run() {
                            //connect(server, port , nickName);
                            try{
                                nickName = nickText.getText().toString();

                                socketOut.println("#sendMC@" + nickName + ":" + msgText.getText().toString());
                                logger("Me:" + msgText.getText().toString());
                            }catch (Exception e){
                                //textView.append(e.toString());
                            }
                        }
                    }.start();

                } else {

                }
                break;
        }
    }

    private void logger(String MSG) {
        tv.append(MSG + "\n");     // 텍스트뷰에 메세지를 더해줍니다.
        sv.fullScroll(ScrollView.FOCUS_DOWN); // 스크롤뷰의 스크롤을 내려줍니다.
    }

    private class MyThread1 extends Thread {
        @Override
        public void run() {
            while (true) {
                try {
                    // InputStream의 값을 읽어와서 data에 저장
                    String data = socketIn.readLine();
                    // Message 객체를 생성, 핸들러에 정보를 보낼 땐 이 메세지 객체를 이용
                    Message msg = myHandler.obtainMessage();
                    msg.obj = data;
                    myHandler.sendMessage(msg);
                }
                catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    };

    private class MyHandler1 extends Handler {
        boolean notRegistered = true;

        @Override
        public void handleMessage(Message msg) {
            String[] TK = msg.obj.toString().split("#");
            try{
                if(TK.length > 0){
                    if(TK[0].compareTo("CCresult") == 0){
                        logger("Guide:" + TK[1] + "");
                    }else if(TK[0].compareTo("register") == 0){
                        if(notRegistered){
                            new Thread() {
                                public void run() {
                                    //connect(server, port , nickName);
                                    try{
                                        nickName = nickText.getText().toString();

                                        socketOut.println("#0@" + nickName + " : " + "@None");

                                        notRegistered = false;
                                    }catch (Exception e){
                                        logger(e.toString());
                                        logger("Register Error!");
                                    }
                                }
                            }.start();
                        }
                    }else{
                        //logger(msg.obj.toString() + " got!");
                    }
                }

            }catch (Exception ex){
                logger(ex.toString());
            }

        }
    };

}
