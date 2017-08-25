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
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import java.util.Arrays;
import java.util.Vector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;


public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private Mat img_input = null;
    private Mat img_result = new Mat();
    private Mat img_HSV = null;

    private Mat template1;
    private Mat template2;
    private Mat template3;
    private Mat template4;
    private Mat template5;

    private Mat sign1;
    private Mat sign2;
    private Mat sign3;
    private Mat sign4;
    private Mat sign5;
    private Mat sign6;
    private Mat sign7;
    private Mat sign_default;

    int CURRENT = -1;
    int CURRENT_SIGN = -1;

    int[] map_resources = {R.drawable.map1, R.drawable.map2, R.drawable.map3, R.drawable.map4, R.drawable.map5};

    int SelectedCord = 0;
    int Code = 0, templateCode = 0;
    int Code1 = 0, Code2 = 0, Code3 = 0, Code4 = 0;
    int[] Codes = new int[4];
    int[] X1 = {2, 4, 12, 2, 6};
    int[] X2 = {6, 4, 12, 2, 4};
    int[] X3 = {4, 6, 12, 2, 6};
    int[] X4 = {3, 2, 12, 3, 5};

    static int[] pathCode = new int[50];
    static String path = "";
    static int Step_Num = 1;

    static int nE = 5, nV = 5;

    static int[] dist = new int[99];
    static boolean[] visit = new boolean[100];

    static final int inf = 1000000;

    int Start_Point = 0;
    int Destination = 0;
    int[] path_int = new int[100];
    int path_Length = 1;

    int[] sign_toilets_directions = {5, 1, 5, 7, 6};

    static int[][] ad  = {
            {inf, 3, 2, 1, 1, 2},
            {3, inf, 5, 5, 2, inf},
            {2, 1, inf, inf, inf, inf},
            {2, 5, inf, inf, inf, inf},
            {1, 1, inf, inf, inf, inf},
            {5, 6, 2, inf, 7, inf} };

    static int[][] sign  = {
            {7, 3, 2, 1, 6, 2},
            {3, 7, 3, 1, 2, 1},
            {6, 5, 7, 1, 5, 1},
            {2, 2, 0, 7, 4, 0},
            {1, 1, 1, 2, 7, 3},
            {1, 3, 2, 1, 0, 7} };

    boolean isConnected = false;
    TextView textView8 = null;
    Spinner spinner = null;
    Button viewButton = null;

    int answerCode = 0;
    Intent intent = null;

    private static final String TAG = "opencv";
    private CameraBridgeViewBase mOpenCvCameraView;

    public native int convertNativeLib(long matAddrInput, long matAddrResult, long template1, long template2,
                                       long template3,  long template4,  long template5);

    static final int PERMISSION_REQUEST_CODE = 1;
    String[] PERMISSIONS  = {"android.permission.CAMERA"};

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

        final AlertDialog.Builder myDialog = new AlertDialog.Builder(  MainActivity.this);
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

    public static String ssp(int start, int end){
        int n = 5; //배열의 최대길이
        int m = 5000; //너무 멀어서 이동하지 못하는 값 (다른수에 비해 충분히 크면 됨)
        int i,j,k=0;
        int s,e,min;
        int [] v = new int[n];
        int [] distance = new int[n]; //누적거리 배열
        int [] via = new int[n];

        int [][] data = ad;

        s = 3;

        e = 1;
        for( j = 0; j < n; j++ )
        {
            v[j] = 0;
            distance[j] = m;
        }

        distance[s-1] = 0;

        for( i=0; i < n; i++ )
        {
            min = m;
            for( j=0; j < n; j++ )
            {
                if( v[j] == 0 && distance[j] < min )
                {
                    k = j;
                    min = distance[j];
                }
            }

            v[k] = 1;

            if(min == m) break;

            for(j = 0; j < n; j++)
            {
                if(distance[j] > distance[k] + data[k][j])
                {
                    distance[j] = distance[k] + data[k][j];
                    via[j]=k;
                }
            }
        }

        int path[] = new int[n];
        int path_cnt=0;
        k = e-1;
        while(true)
        {
            path[path_cnt++]=k;
            if(k == s-1)break;
            k = via[k];
        }

        String result = "";

        System.out.print(" 경로는 : ");

        for(i = path_cnt-1; i >= 1; i--)
        {
            System.out.printf("%d -> ",path[i]+1);
            result  += (path[i]+1) + " ";
        }
        System.out.printf("%d입니다",path[i]+1);
        result  += (path[i]+1);

        return result;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags( WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (!hasPermissions(PERMISSIONS)) { //퍼미션 허가를 했었는지 여부를 확인
            requestNecessaryPermissions(PERMISSIONS);//퍼미션 허가안되어 있다면 사용자에게 요청
        } else {
            //이미 사용자에게 퍼미션 허가를 받음.
        }

        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(0); // front-camera(1),  back-camera(0)
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        intent = getIntent();
        answerCode = intent.getIntExtra("index", 1);

        textView8 = (TextView)findViewById(R.id.textView8);

        Button button = (Button)findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                SelectedCord = Code;

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

                Codes[0] = Code1;
                Codes[1] = Code2;
                Codes[2] = Code3;
                Codes[3] = Code4;

                //textView.setText(Code1 + " " + Code2 + " " + Code3 + " " + Code4);
                CURRENT = calculate_Distance();
                textView8.setText(CURRENT + " , " + Code1 + " " +  Code2 + " " +  Code3 + " " +  Code4);

                if(answerCode == 1){
                    int Point = calculate_Distance();
                    CURRENT_SIGN = sign_toilets_directions[Point];
                }else if(answerCode == 22){
                    if(CURRENT >= 0){
                        int Point = calculate_Distance();

                        if(CURRENT < path_Length)
                            if(path_int[CURRENT + 1] == Point){
                                CURRENT += 1;
                                CURRENT_SIGN = sign[CURRENT - 1][CURRENT];
                            }

                        if(CURRENT == path_Length){
                            CURRENT = -1;
                            CURRENT_SIGN = -1;
                        }

                    }else{
                        Start_Point = calculate_Distance();
                    }
                }



            }
        });

        Button button2 = (Button)findViewById(R.id.button4);
        button2.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){

                try {
                    String s = ssp(Start_Point, Destination);

                    String[] tokens = s.split(" ");

                    path_Length = tokens.length - 1;
                    for(int i = 0; i < tokens.length; i++){
                        path_int[i] = Integer.parseInt(tokens[i]);
                    }

                    CURRENT = 0;
                    CURRENT_SIGN = sign[path_int[0]][path_int[1]];

                    textView8.setText(s);
                }catch (Exception e){
                    textView8.setText(e.getMessage());
                }

                //textView8.setText(s);

                //Intent intent = new Intent(getApplicationContext(), Main3Activity.class);
                //startActivity(intent);

            }
        });

        viewButton = (Button)findViewById(R.id.button8);
        viewButton.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                if(img_input != null){
                    //img_input.release();
                }

                if(img_result != null){
                    //img_result.release();
                }

                Intent intent = new Intent(getApplicationContext(), MainPage.class);

                try{
                    intent.putExtra("markercode", CURRENT);
                    startActivity(intent);
                }catch(Exception e){
                    textView8.setText("Error" + e.getMessage());
                }

                //textView8.setText(s);

                //Intent intent = new Intent(getApplicationContext(), Main3Activity.class);
                //startActivity(intent);

            }
        });

        spinner = (Spinner)findViewById(R.id.spinner);
        String[] arraySpinner = new String[] {
                "중앙홀", "중앙복도", "맞은편 회의실", "이벤트 경기장 입구", "스키 경기장 입구"
        };

        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this,
                android.R.layout.simple_spinner_item, arraySpinner);

        spinner.setAdapter(adapter);

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view,
                                       int position, long id) {
                Destination = position;
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });

        try{
            template1 = Utils.loadResource(MainActivity.this, R.drawable.template1, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            template2 = Utils.loadResource(MainActivity.this, R.drawable.template2, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            template3 = Utils.loadResource(MainActivity.this, R.drawable.template3, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            template4 = Utils.loadResource(MainActivity.this, R.drawable.template4, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            template5 = Utils.loadResource(MainActivity.this, R.drawable.template5, Imgcodecs.CV_LOAD_IMAGE_COLOR);

            sign1 = Utils.loadResource(MainActivity.this, R.drawable.sign1, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            sign2 = Utils.loadResource(MainActivity.this, R.drawable.sign2, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            sign3 = Utils.loadResource(MainActivity.this, R.drawable.sign3, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            sign4 = Utils.loadResource(MainActivity.this, R.drawable.sign4, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            sign5 = Utils.loadResource(MainActivity.this, R.drawable.sign5, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            sign6 = Utils.loadResource(MainActivity.this, R.drawable.sign6, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            sign7 = Utils.loadResource(MainActivity.this, R.drawable.sign7, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            sign_default = Utils.loadResource(MainActivity.this, R.drawable.sign_default, Imgcodecs.CV_LOAD_IMAGE_COLOR);


        }catch(IOException ie){

        }

    }

    public int calculate_Distance(){
        int[] Score = new int[5];

        for(int i = 0; i < 5; i++){
            Score[i] += (Codes[0] - X1[i]) * (Codes[0] - X1[i]);
            Score[i] += (Codes[1] - X2[i]) * (Codes[1] - X2[i]);
            Score[i] += (Codes[2] - X3[i]) * (Codes[2] - X3[i]);
            Score[i] += (Codes[3] - X4[i]) * (Codes[3] - X4[i]);
        }

        int Min = 999;
        int index = -1;

        for(int i = 0; i < 5; i ++){
            if(Min > Score[i]){
                Min = Score[i];
                index =  i;
            }
        }
        templateCode = index;


        return templateCode;
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
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        /*
        if(img_input != null){
            if(!img_input.empty()){
                img_input.release();
            }
        }

        if(img_result != null){
            if(!img_result.empty()){
                img_result.release();
            }
        }
        */

        img_input = inputFrame.rgba();
        //img_result = new Mat();
        //img_HSV = new Mat();

        template5 = sign_default;
        if(CURRENT_SIGN == 1){
           template5 = sign1;
        }else if(CURRENT_SIGN == 2){
            template5 = sign2;
        }else if(CURRENT_SIGN == 3){
            template5 = sign3;
        }else if(CURRENT_SIGN == 4){
            template5 = sign4;
        }else if(CURRENT_SIGN == 5){
            template5 = sign5;
        }else if(CURRENT_SIGN == 6){
            template5 = sign6;
        }else if(CURRENT_SIGN == 7){
            template5 = sign7;
        }

        Code = convertNativeLib(img_input.getNativeObjAddr(), img_result.getNativeObjAddr(),
                template1.getNativeObjAddr(), template2.getNativeObjAddr(), template3.getNativeObjAddr(),
                template4.getNativeObjAddr(), template5.getNativeObjAddr());

        return img_input;
    }

}

