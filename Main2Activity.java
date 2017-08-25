package com.example.administrator.opencvtest;

import android.app.Activity;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.support.v7.app.AlertDialog;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.io.*;
import java.net.*;
import android.app.*;
import android.os.*;
import android.view.*;
import android.widget.*;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;

public class Main2Activity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    WordBag[] bags;

    private Mat img_input;
    private Mat img_result;
    private Mat img_HSV;

    private Mat template1;
    private Mat template2;
    private Mat template3;
    private Mat template4;
    private Mat template5;

    EditText editText;
    Button button;
    Button connectButton;
    TextView textView;

    int activityCode = -1; int answerCode = -1;
    int[] activityCodes;
    int[] categoryCodes;

    int SpinnerCode = -1; int Code = 0;

    int Code1 = 0, Code2 = 0, Code3 = 0, Code4 = 0;
    int NumberOfLabel;

    int SelectedCore = 0;

    boolean isConnected = false;
    int CurrentCode = -1;

    private static final String TAG = "opencv";
    private CameraBridgeViewBase mOpenCvCameraView;

    public native int convertNativeLib(long matAddrInput, long matAddrResult);

    static final int PERMISSION_REQUEST_CODE = 1;
    String[] PERMISSIONS  = {"android.permission.CAMERA"};

    Spinner spinner;

    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("native-lib");
    }

    private boolean hasPermissions(String[] permissions) {
        int ret = 0;
        //스트링 배열에 있는 퍼미션들의 허가 상태 여부 확인
        for (String perms : permissions){
            ret = checkCallingOrSelfPermission(perms);
            if (!(ret == PackageManager.PERMISSION_GRANTED)){
                //퍼미션 허가 안된 경우
                return false;
            }

        }
        //모든 퍼미션이 허가된 경우
        return true;
    }

    private void requestNecessaryPermissions(String[] permissions) {
        //마시멜로( API 23 )이상에서 런타임 퍼미션(Runtime Permission) 요청
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(permissions, PERMISSION_REQUEST_CODE);
        }
    }



    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onRequestPermissionsResult(int permsRequestCode, String[] permissions, int[] grantResults){
        switch(permsRequestCode){

            case PERMISSION_REQUEST_CODE:
                if (grantResults.length > 0) {
                    boolean camreaAccepted = grantResults[0] == PackageManager.PERMISSION_GRANTED;

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {

                        if (!camreaAccepted  )
                        {
                            showDialogforPermission("앱을 실행하려면 퍼미션을 허가하셔야합니다.");
                            return;
                        }else
                        {
                            //이미 사용자에게 퍼미션 허가를 받음.
                        }
                    }
                }
                break;
        }
    }

    private void showDialogforPermission(String msg) {

        final AlertDialog.Builder myDialog = new AlertDialog.Builder(  Main2Activity.this);
        myDialog.setTitle("알림");
        myDialog.setMessage(msg);
        myDialog.setCancelable(false);
        myDialog.setPositiveButton("예", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface arg0, int arg1) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    requestPermissions(PERMISSIONS, PERMISSION_REQUEST_CODE);
                }

            }
        });
        myDialog.setNegativeButton("아니오", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface arg0, int arg1) {
                finish();
            }
        });
        myDialog.show();
    }

    void AnswerWithAction(String str){

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);

        getWindow().addFlags( WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (!hasPermissions(PERMISSIONS)) { //퍼미션 허가를 했었는지 여부를 확인
            requestNecessaryPermissions(PERMISSIONS);//퍼미션 허가안되어 있다면 사용자에게 요청
        } else {
            //이미 사용자에게 퍼미션 허가를 받음.
        }

        spinner = (Spinner)findViewById(R.id.spinner2);
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

        final DBHelper dbHelper = new DBHelper(getApplicationContext(), "MoneyBook.db", null, 1);

        textView = (TextView)findViewById(R.id.textView);
        //textView.setText("asd");
        editText = (EditText)findViewById(R.id.editText);
        button = (Button)findViewById(R.id.button2);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if(true){
                    double Max = 0.5;
                    answerCode = -1;
                    activityCode = -1;

                    String[] token = editText.getText().toString().split(" ");

                    for(int i = 0; i < NumberOfLabel; i++){
                        double temp = 0;

                        for(int a = 0; a < token.length; a++){
                            temp += bags[i].getScore(token[a]);
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
                                Intent intent = new Intent(getApplicationContext(), activity_map.class);
                                intent.putExtra("index", answerCode);
                                intent.putExtra("activity", activityCode);

                                startActivity(intent);
                            }else if(activityCode == 4){
                                Intent intent = new Intent(getApplicationContext(), activity_map.class);
                                intent.putExtra("index", answerCode);
                                intent.putExtra("activity", activityCode);

                                startActivity(intent);
                            }else if(activityCode == 5){
                                Intent intent = new Intent(getApplicationContext(), AnswerActivity.class);
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

        connectButton = (Button)findViewById(R.id.button3);
        connectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //connect
                SelectedCore = Code;

                if(Code == 0){
                    isConnected = false;
                }else{
                    isConnected = true;
                }

                int Cnt = 1000000;
                Code1 = Code / Cnt;
                Code = Code % Cnt;
                Cnt = Cnt / 100;

                Code2 = Code / Cnt;
                Code = Code % Cnt;
                Cnt = Cnt / 100;

                Code3 = Code / Cnt;
                Code = Code % Cnt;
                Cnt = Cnt / 100;

                Code4 = Code / Cnt;
                Code = Code % Cnt;
                Cnt = Cnt / 100;

                textView.setText(Code1 + " " + Code2 + " " + Code3 + " " + Code4);

                if(Code1 + Code2 + Code3 + Code4 > 0){

                }
            }
        });

        try{
            template1 = Utils.loadResource(Main2Activity.this, R.drawable.template1, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            template2 = Utils.loadResource(Main2Activity.this, R.drawable.template2, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            template3 = Utils.loadResource(Main2Activity.this, R.drawable.template3, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            template4 = Utils.loadResource(Main2Activity.this, R.drawable.template4, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
            template5 = Utils.loadResource(Main2Activity.this, R.drawable.template5, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);

        }catch(IOException ie){

        }

        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(0); // front-camera(1),  back-camera(0)
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

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
                NumberOfLabel = Length;

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
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "onResume :: Internal OpenCV library not found.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "onResum :: OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){

        img_input = inputFrame.rgba();
        img_result = new Mat();
        img_HSV = new Mat();

        Code = convertNativeLib(img_input.getNativeObjAddr(), img_result.getNativeObjAddr());
        //textView.setText(" ");

        return img_input;
    }


}



