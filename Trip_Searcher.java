package com.example.administrator.opencvtest;

import android.app.Activity;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.StrictMode;
import android.support.annotation.DrawableRes;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ListView;
import android.widget.RadioButton;
import android.widget.RadioGroup;
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
import java.util.ArrayList;
import java.util.List;

public class Trip_Searcher extends AppCompatActivity {

    String[] Icon_SRCs = new String[78];
    String[] Icon_SRCs_Food = new String[27];

    private ListView mListView;

    int Position = 0;
    ImageButton searchButton = null;
    Button cancelButton = null;

    RadioButton isFoodButton = null;
    RadioButton isTripButton = null;
    RadioGroup radioGroup = null;

    public Activity myActivity = this;

    EditText searchForm = null;
    EditText addInfo = null;
    TextView textView = null;

    String[] Words_Food = null;
    float[][] Answer_Label_Food = null;

    String[] Words = null;
    float[][] Answer_Label = null;

    float[][] Trip_Label = new float[78][28];
    String[] Trip_Words = new String[78];

    float[][] Food_Label = new float[27][18];
    String[] Food_Words = new String[27];

    int[] Min_Index = new int[5];
    float Words_Count = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_trip__searcher);
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

        for(int i = 1; i < 79; i++){
            Icon_SRCs[i - 1] = "a" + i;
        }

        for(int i = 1; i < 27; i++){
            Icon_SRCs_Food[i - 1] = "b" + i;
        }

        try{

            // getResources().openRawResource()로 raw 폴더의 원본 파일을 가져온다.
            // txt 파일을 InpuStream에 넣는다. (open 한다)
            InputStream in = getResources().openRawResource(R.raw.answerlabel);

            if(in != null){

                InputStreamReader stream = new InputStreamReader(in, "utf-8");
                BufferedReader buffer = new BufferedReader(stream);

                String read;
                StringBuilder sb = new StringBuilder("");

                for(int i = 0; i < 78; i++){
                    Trip_Words[i] = buffer.readLine();
                    String[] TK = buffer.readLine().split("@");

                    for(int j = 0; j < 28; j++){
                        Trip_Label[i][j] = Integer.parseInt(TK[j]);
                    }
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
            InputStream in = getResources().openRawResource(R.raw.answerlabel_food);

            if(in != null){

                InputStreamReader stream = new InputStreamReader(in, "utf-8");
                BufferedReader buffer = new BufferedReader(stream);

                String read;
                StringBuilder sb = new StringBuilder("");

                for(int i = 0; i < 27; i++){
                    Food_Words[i] = buffer.readLine();
                    String[] TK = buffer.readLine().split("@");

                    for(int j = 0; j < 18; j++){
                        Food_Label[i][j] = Integer.parseInt(TK[j]);
                    }
                }

            }else{
                //textView.setText("Read Error ");
            }

        }catch(Exception e){
            e.printStackTrace();
            //textView.setText(e.hashCode() + "Read Error " + e.getMessage());
        }
        ////////// Dataset Read

        try{

            // getResources().openRawResource()로 raw 폴더의 원본 파일을 가져온다.
            // txt 파일을 InpuStream에 넣는다. (open 한다)
            InputStream in = getResources().openRawResource(R.raw.tripwordsdatafile);

            if(in != null){

                InputStreamReader stream = new InputStreamReader(in, "utf-8");
                BufferedReader buffer = new BufferedReader(stream);

                String read;
                StringBuilder sb = new StringBuilder("");

                int Length = Integer.parseInt(buffer.readLine());
                Words = new String[Length];
                Answer_Label = new float[Length][28];
                for(int i = 0; i < Length; i++){
                    for(int j = 0; j < 28; j++){
                        Answer_Label[i][j] = 0;
                    }
                }

                for(int i = 0; i < Length; i++){
                    String[] TK = buffer.readLine().split("#");
                    String[] Label_TK = TK[1].split("@");

                    Words[i] = TK[0];
                    for(int j = 0; j < 28; j++){
                        Answer_Label[i][j] = Integer.parseInt(Label_TK[j]);
                    }
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
            InputStream in = getResources().openRawResource(R.raw.fooddataset);

            if(in != null){

                InputStreamReader stream = new InputStreamReader(in, "utf-8");
                BufferedReader buffer = new BufferedReader(stream);

                String read;
                StringBuilder sb = new StringBuilder("");

                int Length = Integer.parseInt(buffer.readLine());
                Words_Food = new String[Length];
                Answer_Label_Food = new float[Length][18];
                for(int i = 0; i < Length; i++){
                    for(int j = 0; j < 18; j++){
                        Answer_Label_Food[i][j] = 0;
                    }
                }

                for(int i = 0; i < Length; i++){
                    String[] TK = buffer.readLine().split("#");
                    String[] Label_TK = TK[1].split("@");

                    Words_Food[i] = TK[0];
                    for(int j = 0; j < 18; j++){
                        Answer_Label_Food[i][j] = Integer.parseInt(Label_TK[j]);
                    }
                }
            }else{
                textView.setText("Read Error ");
            }

        }catch(Exception e){
            e.printStackTrace();
            textView.setText(e.hashCode() + "Read Error " + e.getMessage());
        }
        ////////// Dataset Read

        Intent intent = getIntent();
        Position = intent.getIntExtra("Position", -1);

        searchButton = (ImageButton)findViewById(R.id.imageButton4);
        isFoodButton = (RadioButton)findViewById(R.id.radioButton2);
        isTripButton = (RadioButton)findViewById(R.id.radioButton3);
        textView = (TextView)findViewById(R.id.textView10);

        searchForm = (EditText)findViewById(R.id.editText3);
        addInfo = (EditText)findViewById(R.id.info_edittext);

        searchButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if(isTripButton.isChecked()){
                    String[] Searched_Words = searchForm.getText().toString().split(" ");
                    float[] Score = new float[28];

                    for(int k = 0; k < 28; k++){
                        Score[k] = 0;
                    }

                    Words_Count = 0;

                    for(int k = 0; k < Searched_Words.length; k++){

                        for(int i = 0; i < Words.length; i++){
                            if(Words[i].compareTo(Searched_Words[k]) == 0){
                                for(int j = 0; j < 28; j++){
                                    Score[j] += Answer_Label[i][j];
                                }

                                i = Words.length;
                                Words_Count++;
                            }
                        }

                    }

                    textView.append(" " + Words_Count + ": ");

                    if(Words_Count == 0){
                        textView.setText("No Search Result!");
                    }else{
                        for(int j = 0; j < 28; j++){
                            textView.append(Score[j] + " ");
                            Score[j] = Score[j] / Words_Count;
                        }

                        float[] Predicted_Scores = new float[78];

                        for(int i = 0; i < 78; i++){
                            Predicted_Scores[i] = 0;

                            for(int j = 0; j < 28; j++){
                                Predicted_Scores[i] += (Trip_Label[i][j] - Score[j]) * (Trip_Label[i][j] - Score[j]);
                            }
                        }

                        float[] Min = new float[5];
                        for(int i = 0; i < 5; i++){
                            Min[i] = 999;
                        }

                        for(int k = 0; k < 5; k++){
                            for(int i = 0; i < 78; i++){
                                if(Predicted_Scores[i] < Min[k]){
                                    Min[k] = Predicted_Scores[i];
                                    Min_Index[k] = i;
                                }
                            }

                            Predicted_Scores[Min_Index[k]] = 999;
                        }

                        textView.setText(Trip_Words[Min_Index[0]] + " Searched!");
                        for(int i = 1; i < 5; i++){
                            textView.append(Trip_Words[Min_Index[i]] + " Searched! ");
                        }

                        dataSetting();
                    }
                }else {
                    //search Food
                    String[] Searched_Words = searchForm.getText().toString().split(" ");
                    float[] Score = new float[18];

                    for (int k = 0; k < 18; k++) {
                        Score[k] = 0;
                    }

                    Words_Count = 0;

                    for (int k = 0; k < Searched_Words.length; k++) {

                        for (int i = 0; i < Words_Food.length; i++) {
                            if (Words_Food[i].compareTo(Searched_Words[k]) == 0) {
                                for (int j = 0; j < 18; j++) {
                                    Score[j] += Answer_Label_Food[i][j];
                                }

                                i = Words_Food.length;
                                Words_Count++;
                            }
                        }

                    }

                    textView.append(" " + Words_Count + ": ");

                    if (Words_Count == 0) {
                        textView.setText("No Search Result!");
                    } else {
                        for (int j = 0; j < 18; j++) {
                            textView.append(Score[j] + " ");
                            Score[j] = Score[j] / Words_Count;
                        }

                        float[] Predicted_Scores = new float[27];

                        for (int i = 0; i < 27; i++) {
                            Predicted_Scores[i] = 0;

                            for (int j = 0; j < 18; j++) {
                                Predicted_Scores[i] += (Trip_Label[i][j] - Score[j]) * (Trip_Label[i][j] - Score[j]);
                            }
                        }

                        float[] Min = new float[5];
                        for (int i = 0; i < 5; i++) {
                            Min[i] = 999;
                        }

                        for (int k = 0; k < 5; k++) {
                            for (int i = 0; i < 27; i++) {
                                if (Predicted_Scores[i] < Min[k]) {
                                    Min[k] = Predicted_Scores[i];
                                    Min_Index[k] = i;
                                }
                            }

                            Predicted_Scores[Min_Index[k]] = 999;
                        }

                        textView.setText(Food_Words[Min_Index[0]] + " Searched!");
                        for (int i = 1; i < 5; i++) {
                            textView.append(Food_Words[Min_Index[i]] + " Searched! ");
                        }

                        dataSetting();

                    }
                }

            }
        });

        mListView = (ListView)findViewById(R.id.list_view_searcher);
    }

    private void dataSetting(){

        MyAdapter mMyAdapter = new MyAdapter();

        //Icon_SRCs[Min_Index[i]]
        if(isTripButton.isChecked()){
            for (int i=0; i<5; i++) {
                int id = getResources().getIdentifier(Icon_SRCs[Min_Index[i]], "drawable", getPackageName());
                Drawable drawable = getResources().getDrawable(id);

                mMyAdapter.addItem(drawable, Trip_Words[Min_Index[i]], "contents_" + i);
            }
        }else{
            for (int i=0; i<5; i++) {
                int id = getResources().getIdentifier(Icon_SRCs_Food[Min_Index[i]], "drawable", getPackageName());
                Drawable drawable = getResources().getDrawable(id);

                mMyAdapter.addItem(drawable, Food_Words[Min_Index[i]], "contents_" + i);
            }
        }

        /* 리스트뷰에 어댑터 등록 */
        mListView.setAdapter(mMyAdapter);
        mListView.setOnItemClickListener(itemClickListenerList);
    }

    private AdapterView.OnItemClickListener itemClickListenerList = new AdapterView.OnItemClickListener()
    {
        public void onItemClick(AdapterView<?> adapterView, View clickedView, int pos, long id)
        {
            Intent intent = new Intent();
            intent.putExtra("search_result", Min_Index[pos]);
            intent.putExtra("position", Position);
            intent.putExtra("addinfo", addInfo.getText().toString());
            if(!isTripButton.isChecked()){
                intent.putExtra("Code", -1);
            }

            setResult(RESULT_OK, intent);
            finish();
        }
    };

}
