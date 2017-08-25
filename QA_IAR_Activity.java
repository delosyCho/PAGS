package com.example.administrator.opencvtest;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

public class QA_IAR_Activity extends AppCompatActivity {

    ImageButton QuestionButton = null;
    TextView AnswerView = null;
    EditText Edit_Question = null;

    Button gochatbutton = null;

    //  TCP연결 관련
    private Socket clientSocket;
    private BufferedReader socketIn;
    private PrintWriter socketOut;
    private int port = 17755;
    private final String ip = "192.168.0.8";
    private MyHandler myHandler;
    private MyThread myThread;

    String Line = "";
    String Message_Got = "";

     private class MyThread extends Thread {
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
    }

    private class MyHandler extends Handler {
        @Override
        public void handleMessage(Message msg) {
            String[] TK = msg.obj.toString().split("#");
            if(TK[0].compareTo("result") == 0){

            }

            AnswerView.setText(msg.obj.toString() + " got!");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_qa__iar_);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                        .setAction("Action", null).show();
            }
        });

        gochatbutton = (Button)findViewById(R.id.gochatbutton);
        gochatbutton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), Main3Activity.class);
                //intent.putExtra("question", Question);
                startActivity(intent);
            }
        });

        new Thread() {
            public void run() {
                //connect(server, port , nickName);

                try {
                    clientSocket = new Socket(ip, port);
                    socketIn = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
                    socketOut = new PrintWriter(clientSocket.getOutputStream(), true);
                } catch (Exception e) {
                    //textView.setText(e.toString());
                }
            }
        }.start();

        myHandler = new MyHandler();
        myThread = new MyThread();
        myThread.start();

        QuestionButton = (ImageButton)findViewById(R.id.imageButton5);
        QuestionButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder().permitAll().build();
                //StrictMode.setThreadPolicy(policy);

                new Thread() {
                    public void run() {
                        //connect(server, port , nickName);
                        try{
                            socketOut.println(Line);
                        }catch (Exception e){
                            //textView.append(e.toString());
                        }
                    }
                }.start();
            }
        });

        AnswerView = (TextView)findViewById(R.id.textView11);
        Edit_Question = (EditText)findViewById(R.id.editText5);
    }

}
