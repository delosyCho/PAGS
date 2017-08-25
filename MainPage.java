package com.example.administrator.opencvtest;

import android.content.Intent;
import android.media.Image;
import android.os.Bundle;
import android.app.Activity;
import android.widget.ImageView;

public class MainPage extends Activity {

    int[] map_resources = {R.drawable.map1, R.drawable.map2, R.drawable.map3, R.drawable.map4, R.drawable.map5};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_page);

        Intent intent = getIntent();
        intent = getIntent();
        int markerCode = intent.getIntExtra("markercode", 1);

        ImageView imageView = (ImageView)findViewById(R.id.imageView2);
        imageView.setImageResource(map_resources[markerCode]);

    }

}
