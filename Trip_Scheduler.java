package com.example.administrator.opencvtest;

import android.content.Context;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ListView;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class Trip_Scheduler extends AppCompatActivity {

    int List_Length = 0;

    int REQUEST_ACT = 0;

    Button button = null;
    Button add_button = null;

    int default_Blank = 20;
    int number_of_Blanks = default_Blank;

    String[] Titles = new String[default_Blank];
    String[] Contents = new String[default_Blank];
    String[] Image_Drawables = new String[default_Blank];
    int[] Year = new int[default_Blank];
    int[] Month = new int[default_Blank];
    int[] Day = new int[default_Blank];
    int[] Hour = new int[default_Blank];
    int[] Minute = new int[default_Blank];

    private ListView mListView;

    float[][] Trip_Label = new float[78][28];
    String[] Trip_Words = new String[78];

    float[][] Food_Label = new float[27][18];
    String[] Food_Words = new String[27];

    String default_Image = "buttonblank";
    String[] Icon_SRCs = new String[78];
    String[] Icon_SRCs_Food = new String[27];

    RadioButton schedile_Radio = null;
    RadioButton date_Radio = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_trip__scheduler);
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

        schedile_Radio = (RadioButton)findViewById(R.id.scheduleradio);
        date_Radio = (RadioButton)findViewById(R.id.dataradio);

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
                //textView.setText("Read Error ");
            }

        }catch(Exception e){
            e.printStackTrace();
            //textView.setText(e.hashCode() + "Read Error " + e.getMessage());
        }

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

        for(int i = 0; i < default_Blank; i++){
            Titles[i] = "Blank";
            Contents[i] = " ";
            Image_Drawables[i] = default_Image;
            Year[i] = -1;
            Month[i] = -1;
            Hour[i] = -1;
            Minute[i] = -1;
        }

        button = (Button)findViewById(R.id.button11);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                String str = "STR" ;

                try {
                    // open file.
                    FileOutputStream fos = openFileOutput("schedule.txt",
                            Context.MODE_PRIVATE);

                    fos.write(new String((List_Length + 1) + "#").getBytes());

                    for(int i = 0; i < List_Length + 1; i++){
                        String Date_LIne = Year[i] + "/" + Month[i] + "/" + Day[i] + "/" + Hour[i] + "/" + Minute[i];

                        str = Titles[i] + "@" + Contents[i] + "@" + Image_Drawables[i] + "@" + Date_LIne + "#";
                        fos.write(str.getBytes());
                    }

                    fos.close();

                } catch (Exception e) {
                    button.setText(e.toString());
                    e.printStackTrace() ;
                }

            }
        });

        add_button = (Button)findViewById(R.id.button14);
        add_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String[] TitlesT = new String[number_of_Blanks + 5];
                String[] ContentsT = new String[number_of_Blanks + 5];
                String[] Image_DrawableT = new String[number_of_Blanks + 5];
                for(int i = 0; i < number_of_Blanks; i++){
                    TitlesT[i] = Titles[i];
                    ContentsT[i] = Contents[i];
                    Image_DrawableT[i] = Image_Drawables[i];
                }

                for(int i = number_of_Blanks; i < number_of_Blanks + 5; i++){
                    TitlesT[i] = "Blank";
                    ContentsT[i] = " ";
                    Image_DrawableT[i] = default_Image;
                }

                number_of_Blanks += 5;

                Titles = TitlesT;
                Contents = ContentsT;

                Image_Drawables = Image_DrawableT;

                dataSetting();

            }
        });

        mListView = (ListView)findViewById(R.id.list_view1);
        dataSetting();
    }

    private void dataSetting(){

        MyAdapter mMyAdapter = new MyAdapter();

        for (int i=0; i<number_of_Blanks; i++) {
            String Content = Contents[i];
            if(Year[i] != -1){
                Content += '\n';
                Content += Year[i] + "." + Month[i] + "." + Day[i] + ". " + Hour[i] + ":" + Minute[i];
            }

            int id = getResources().getIdentifier(Image_Drawables[i], "drawable", getPackageName());
            Drawable drawable = getResources().getDrawable(id);
            mMyAdapter.addItem(drawable, Titles[i], Content);
        }

        /* 리스트뷰에 어댑터 등록 */
        mListView.setAdapter(mMyAdapter);
        mListView.setOnItemClickListener(itemClickListenerList);
    }

    private AdapterView.OnItemClickListener itemClickListenerList = new AdapterView.OnItemClickListener()
    {
        public void onItemClick(AdapterView<?> adapterView, View clickedView, int pos, long id)
        {
            if(schedile_Radio.isChecked()){
                Intent intent = new Intent(getApplicationContext(), Trip_Searcher.class);
                intent.putExtra("Position", pos);
                startActivityForResult(intent, REQUEST_ACT);
            }else{
                Intent intent = new Intent(getApplicationContext(), Schedule_Date_Setter.class);
                intent.putExtra("Position", pos);
                startActivityForResult(intent, REQUEST_ACT);
            }
        }
    };


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(schedile_Radio.isChecked()){
            if (resultCode != RESULT_OK) {
                //Toast.makeText(MainActivity.this, "결과가 성공이 아님.", Toast.LENGTH_SHORT).show();
                return;
            }else{
                if (requestCode == REQUEST_ACT) {
                    int IsTrip = data.getIntExtra("Code", 1);

                    if(IsTrip == 1){
                        String resultMsg = data.getStringExtra("result_msg");

                        int position = data.getIntExtra("position", -1);
                        int result = data.getIntExtra("search_result", -1);

                        if(position != -1){
                            Titles[position] = Trip_Words[result];
                            Image_Drawables[position] = Icon_SRCs[result];
                            String content = data.getStringExtra("addinfo");
                            Contents[position] = content;

                            dataSetting();

                            if(List_Length < position){
                                List_Length = position;
                            }
                        }
                    }else{
                        String resultMsg = data.getStringExtra("result_msg");

                        int position = data.getIntExtra("position", -1);
                        int result = data.getIntExtra("search_result", -1);

                        if(position != -1){
                            Titles[position] = Food_Words[result];
                            Image_Drawables[position] = Icon_SRCs_Food[result];
                            String content = data.getStringExtra("addinfo");
                            Contents[position] = content;

                            dataSetting();

                            if(List_Length < position){
                                List_Length = position;
                            }
                        }
                    }

                }else {

                }
            }

        }else{
            if (resultCode != RESULT_OK) {
                //Toast.makeText(MainActivity.this, "결과가 성공이 아님.", Toast.LENGTH_SHORT).show();
                return;
            }else{
                if (requestCode == REQUEST_ACT) {
                    String resultMsg = data.getStringExtra("result_msg");

                    int position = data.getIntExtra("position", -1);
                    int year = data.getIntExtra("year", -1);
                    int month = data.getIntExtra("month", -1);
                    int day = data.getIntExtra("day", -1);
                    int hour = data.getIntExtra("time", -1);
                    int minute = data.getIntExtra("minute", -1);

                    if(position != -1){
                        Year[position] = year;
                        Month[position] = month;
                        Day[position] = day;
                        Hour[position] = hour;
                        Minute[position] = minute;

                        dataSetting();

                        if(List_Length < position){
                            List_Length = position;
                        }
                    }

                } else {

                }
            }

        }


    }

}

