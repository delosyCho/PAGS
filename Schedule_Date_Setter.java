package com.example.administrator.opencvtest;

import android.content.Intent;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.Button;
import android.widget.DatePicker;
import android.widget.TimePicker;

public class Schedule_Date_Setter extends AppCompatActivity {

    TimePicker timePicker = null;
    DatePicker datePicker = null;
    Button button = null;

    int Position = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_schedule__date__setter);
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

        Intent intent2 = getIntent();
        Position = intent2.getIntExtra("Position", -1);

        timePicker = (TimePicker)findViewById(R.id.timepicker);
        datePicker = (DatePicker)findViewById(R.id.datepicker);
        button = (Button)findViewById(R.id.datesetbutton);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.putExtra("position", Position);
                intent.putExtra("year", datePicker.getYear());
                intent.putExtra("month", datePicker.getMonth());
                intent.putExtra("day", datePicker.getDayOfMonth());
                intent.putExtra("time", timePicker.getHour());
                intent.putExtra("minute", timePicker.getMinute());

                setResult(RESULT_OK, intent);
                finish();
            }
        });
    }

}
