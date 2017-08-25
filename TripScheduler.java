package com.example.administrator.opencvtest;

import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;

import org.w3c.dom.Text;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;

public class TripScheduler extends AppCompatActivity {

    int List_Length = 0;
    int default_Blank = 20;

    String[] Titles = new String[default_Blank];
    String[] Contents = new String[default_Blank];
    String[] Image_Drawables = new String[default_Blank];

    int[] Year = new int[default_Blank];
    int[] Month = new int[default_Blank];
    int[] Day = new int[default_Blank];
    int[] Hour = new int[default_Blank];
    int[] Minute = new int[default_Blank];

    private ListView mListView;

    Button setScheduleButton = null;
    Button onAlaramButton = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_trip_scheduler);
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

        setScheduleButton = (Button)findViewById(R.id.setschedule);
        setScheduleButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), Trip_Scheduler.class);
                startActivity(intent);
            }
        });

        onAlaramButton = (Button)findViewById(R.id.onalaram);
        onAlaramButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(List_Length > 0){
                    for(int i = 0;  i < List_Length; i++){
                        NotificationManager notificationManager= (NotificationManager)TripScheduler.this.getSystemService(TripScheduler.this.NOTIFICATION_SERVICE);
                        Intent intent1 = new Intent(TripScheduler.this.getApplicationContext(),MainActivity.class); //인텐트 생성.

                        Notification.Builder builder = new Notification.Builder(getApplicationContext());
                        intent1.addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP| Intent.FLAG_ACTIVITY_CLEAR_TOP);//현재 액티비티를 최상으로 올리고, 최상의 액티비티를 제외한 모든 액티비티를없앤다.

                        PendingIntent pendingNotificationIntent = PendingIntent.getActivity( TripScheduler.this,0, intent1,PendingIntent.FLAG_UPDATE_CURRENT);
                        //PendingIntent는 일회용 인텐트 같은 개념입니다.
                        //FLAG_UPDATE_CURRENT - > 만일 이미 생성된 PendingIntent가 존재 한다면, 해당 Intent의 내용을 변경함.
                        //FLAG_CANCEL_CURRENT - .이전에 생성한 PendingIntent를 취소하고 새롭게 하나 만든다.
                        // FLAG_NO_CREATE -> 현재 생성된 PendingIntent를 반환합니다.
                        // FLAG_ONE_SHOT - >이 플래그를 사용해 생성된 PendingIntent는 단 한번밖에 사용할 수 없습니다
                        int id = getResources().getIdentifier(Image_Drawables[i], "drawable", getPackageName());

                        builder.setSmallIcon(id).setTicker(Titles[i]).setWhen(System.currentTimeMillis())
                                .setNumber(1).setContentTitle(Titles[i]).setContentText(Contents[i])
                                .setDefaults(Notification.DEFAULT_SOUND | Notification.DEFAULT_VIBRATE).setContentIntent(pendingNotificationIntent).setAutoCancel(true).setOngoing(true);
                        //해당 부분은 API 4.1버전부터 작동합니다.
    //setSmallIcon - > 작은 아이콘 이미지
    //setTicker - > 알람이 출력될 때 상단에 나오는 문구.
    //setWhen -> 알림 출력 시간.
    //setContentTitle-> 알림 제목
    //setConentText->푸쉬내용

                        notificationManager.notify(1, builder.build()); // Notification send

                    }
                }
            }
        });

        String str = "";
        String[] lines = null;

        try {
            FileInputStream fis = openFileInput("schedule.txt");
            byte[] buffer = new byte[fis.available()];
            fis.read(buffer);
            str = new String(buffer);
            lines = str.split("#");

            // read 1 char from file.
            List_Length = Integer.parseInt(lines[0]);
            Titles = new String[List_Length];
            Contents = new String[List_Length];
            Image_Drawables = new String[List_Length];
            Year = new int[List_Length];
            Month = new int[List_Length];
            Day = new int[List_Length];
            Hour = new int[List_Length];
            Minute = new int[List_Length];

            for(int i = 0; i < List_Length; i++){
                String[] TK = lines[i + 1].split("@");
                String[] TK2 = TK[3].split("/");

                Titles[i] = TK[0];
                Contents[i] = TK[1];
                Image_Drawables[i] = TK[2];

                Year[i] = Integer.parseInt(TK2[0]);
                Month[i] = Integer.parseInt(TK2[1]);
                Day[i] = Integer.parseInt(TK2[2]);
                Hour[i] = Integer.parseInt(TK2[3]);
                Minute[i] = Integer.parseInt(TK2[4]);
            }

            fis.close();

            mListView = (ListView)findViewById(R.id.mylistview);
            dataSetting();
        } catch (Exception e) {
            TextView tv = (TextView)findViewById(R.id.textView12);
            tv.append(str);
            tv.append(e.toString());
            e.printStackTrace() ;
        }

    }

    private void dataSetting(){

        MyAdapter mMyAdapter = new MyAdapter();


        for (int i=0; i<List_Length; i++) {
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

        }
    };

}
