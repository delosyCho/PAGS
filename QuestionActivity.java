package com.example.administrator.opencvtest;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.speech.tts.TextToSpeech;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.SurfaceView;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.Utils;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class QuestionActivity extends AppCompatActivity implements TextToSpeech.OnInitListener{

    private TextToSpeech myTTS;
    Thread readyThread;

    EditText editText;
    Button button;
    Button chatButton;

    Button connectButton;
    TextView textView;

    String Question = "";
    ImageView mascotImageView = null;

    int activityCode = -1; int answerCode = -1;
    int[] activityCodes;
    int[] categoryCodes;

    int SpinnerCode = -1; int Code = 0;

    int Code1 = 0, Code2 = 0, Code3 = 0, Code4 = 0;
    int NumberOfLabel;

    int SelectedCore = 0;

    boolean isConnected = false;
    int CurrentCode = -1;

    WordBag[] bags;
    Spinner spinner;

    int cnt = 0;
    int CurrentImageCode = 0;

    int[] mascot_Image_reousrce = {R.drawable.td_1, R.drawable.td_2, R.drawable.td_3, R.drawable.td_4, R.drawable.td_5};

    private MyHandler myHandler = null;
    private MyThread myThread = null;

    class MyThread extends Thread {
        int CurrentCode = 0;
        int fullCount = 0;
        @Override
        public void run() {
            while (fullCount < 58) {
                try {
                    this.sleep(50);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                CurrentCode++;
                if(CurrentCode == 5){
                    CurrentCode = 0;
                    fullCount++;
                }

                Message msg = new Message();
                msg.what = 0;
                msg.arg1 = CurrentCode;

                try {
                    myHandler.sendMessage(msg);
                }
                catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    class MyHandler extends Handler {
        @Override
        public void handleMessage(Message msg) {
            if(msg.what == 0){
                mascotImageView.setImageResource(mascot_Image_reousrce[msg.arg1]);
                //textView.setText("" + msg.arg1);
            }

        }
    }

    private void runThread(){
        runOnUiThread (new Thread(new Runnable() {
            public void run() {
                while(cnt++ < 1000){
                    textView.setText("#"+cnt);
                    try {
                        Thread.sleep(300);
                    }
                    catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }));
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.content_question);
        //Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        //setSupportActionBar(toolbar);

        mascotImageView = (ImageView)findViewById(R.id.imageView);

        myHandler = new MyHandler();
        myThread = new MyThread();
        myThread.start();

        myTTS = new TextToSpeech(this, this);

        spinner = (Spinner)findViewById(R.id.spinner5);
        String[] arraySpinner = new String[] {
                "경기 일정", "위치", "교통", "숙박", "먹거리", "즐길 거리", "긴급", "기타 등등"
        };

        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this,
                android.R.layout.simple_spinner_item, arraySpinner);
        spinner.setAdapter(adapter);

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view,
                                       int position, long id) {
                SpinnerCode = position;
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        textView = (TextView)findViewById(R.id.textView6);
        //textView.setText("asd");
        editText = (EditText)findViewById(R.id.editText2);
        button = (Button)findViewById(R.id.button5);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {


                if(true){
                    double Max = 0.5;
                    answerCode = -1;
                    activityCode = -1;

                    Question = editText.getText().toString();
                    String[] token = editText.getText().toString().split(" ");
                    String test = "";
                    //button.setText(token[0] + bags[19].getScore(token[0]));
                    for(int i = 0; i < NumberOfLabel; i++){
                        double temp = 0;

                        for(int a = 0; a < token.length; a++){
                            temp += bags[i].getScore(token[a]);
                        }

                        if(SpinnerCode == i){
                            temp = temp * 10;
                        }

                        if(Max < temp){
                            Max = temp;
                            answerCode = i;
                        }
                    }

                    if(answerCode != -1){
                        int category = 0;

                        activityCode = activityCodes[answerCode];
                        category = categoryCodes[answerCode];

                        if(SpinnerCode == category){

                            if(activityCode == 0){
                                Intent intent = new Intent(getApplicationContext(), AnswerActivity.class);
                                intent.putExtra("index", answerCode);
                                intent.putExtra("activity", activityCode);

                                startActivity(intent);
                            }else if(activityCode == 1){
                                Intent intent = new Intent(getApplicationContext(), AnswerActivity.class);
                                intent.putExtra("index", answerCode);
                                intent.putExtra("activity", activityCode);

                                startActivity(intent);
                            }else if(activityCode == 2){
                                Intent intent = new Intent(getApplicationContext(), activity_map.class);
                                intent.putExtra("index", answerCode);
                                intent.putExtra("activity", activityCode);

                                startActivity(intent);
                            }else if(activityCode == 3){
                                Intent intent = new Intent(getApplicationContext(), AnswerActivity.class);
                                intent.putExtra("index", answerCode);
                                intent.putExtra("activity", activityCode);

                                startActivity(intent);
                            }else if(activityCode == 4){
                                Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                                intent.putExtra("index", answerCode);
                                intent.putExtra("activity", activityCode);

                                startActivity(intent);
                            }else if(activityCode == 5){
                                Intent intent = new Intent(getApplicationContext(), activity_map.class);
                                intent.putExtra("index", answerCode);
                                intent.putExtra("activity", activityCode);

                                startActivity(intent);
                            }

                            textView.setText("Spinner " + SpinnerCode + " " + category + " answer: " + answerCode + " " + activityCode);

                        }else{
                            textView.setText("Spinner " + SpinnerCode + " " + category + " answer: " + answerCode + " " + activityCode);
                        }
                    }

                }else{

                }

            }
        });

        chatButton = (Button)findViewById(R.id.button7);
        chatButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), QA_IAR_Activity.class);
                intent.putExtra("question", Question);
                startActivity(intent);
            }
        });

        try{

            // getResources().openRawResource()로 raw 폴더의 원본 파일을 가져온다.
            // txt 파일을 InpuStream에 넣는다. (open 한다)
            InputStream in = getResources().openRawResource(R.raw.paigs_wordbag);

            if(in != null){

                InputStreamReader stream = new InputStreamReader(in, "utf-8");
                BufferedReader buffer = new BufferedReader(stream);

                String read;
                StringBuilder sb = new StringBuilder("");

                int Length = Integer.parseInt(buffer.readLine());
                NumberOfLabel = 24;

                bags = new WordBag[Length];

                for(int i = 0; i < Length; i++){
                    String s = "";
                    //textView.setText("" + i);
                    s = buffer.readLine();
                    //textView.setText("aa" + s);
                    bags[i] = new WordBag(s);
                    //textView.setText("Detected: " + s);
                }
            }else{
                textView.setText("Read Error ");
            }

        }catch(Exception e){
            e.printStackTrace();
            textView.setText(e.hashCode() + "Read Error " + e.getMessage());
        }
        ////////// Dataset Read

        try{

            // getResources().openRawResource()로 raw 폴더의 원본 파일을 가져온다.
            // txt 파일을 InpuStream에 넣는다. (open 한다)
            InputStream in = getResources().openRawResource(R.raw.paigs_answer_dataset);

            activityCodes = new int[NumberOfLabel];
            categoryCodes = new int[NumberOfLabel];

            if(in != null){

                InputStreamReader stream = new InputStreamReader(in, "utf-8");
                BufferedReader buffer = new BufferedReader(stream);

                String read;
                StringBuilder sb = new StringBuilder("");

                for(int i = 0; i < NumberOfLabel; i++){
                    String s = buffer.readLine();
                    String[] tokens = s.split(":");

                    activityCodes[i] = Integer.parseInt(tokens[3]);
                    categoryCodes[i] = Integer.parseInt(tokens[2]);

                }
            }else{
                textView.setText("Read Error ");
            }

        }catch(Exception e){
            e.printStackTrace();
            //textView.setText(e.hashCode() + "Database Error " + e.getMessage());
        }
        /////////// Answer Set Read

    }

    @Override
    public void onInit(int status) {
        textView.setText("Speech");
        //runThread();

        String myText1 = "안녕하세요 평창 올림픽 도우미 반다비입니다";
        String myText2 = "원하시는 질문을 입력하신 뒤 엔터 버튼을 클릭해주세요";
        String myText3 = "생각한 답변이 안나오면 밑의 채팅 버튼을 클릭하셔서 직원에게 도움을 청해보세요";

        myTTS.speak(myText1, TextToSpeech.QUEUE_FLUSH, null);
        myTTS.speak(myText2, TextToSpeech.QUEUE_ADD, null);
        myTTS.speak(myText3, TextToSpeech.QUEUE_ADD, null);
        textView.setText("Complete");

    }
}
