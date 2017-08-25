package com.example.administrator.opencvtest;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageButton;

public class Main extends AppCompatActivity {

    WordBag[] bags;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main4);

        ImageButton button2;
        button2 = (ImageButton)findViewById(R.id.imageButton19);
        button2.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){

                Intent intent = new Intent(getApplicationContext(), MainActivity.class);
                startActivity(intent);

            }
        });

        ImageButton button3;
        button3 = (ImageButton)findViewById(R.id.imageButton20);
        button3.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){

                Intent intent = new Intent(getApplicationContext(), QuestionActivity.class);
                startActivity(intent);

            }
        });

        ImageButton button4;
        button4 = (ImageButton)findViewById(R.id.imageButton18);
        button4.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){

                Intent intent = new Intent(getApplicationContext(), TripScheduler.class);
                startActivity(intent);

            }
        });

    }
}
