package com.example.administrator.opencvtest;

import android.app.FragmentManager;
import android.content.Intent;
import android.net.Uri;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.MapFragment;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.model.LatLng;
import com.google.android.gms.maps.model.Marker;
import com.google.android.gms.maps.model.MarkerOptions;

import org.w3c.dom.Text;


public class activity_map extends AppCompatActivity
        implements OnMapReadyCallback {

    int activityCode = -1;
    Intent intent = null;

    Marker BusTerminalMarker = null;
    Marker BusStationMarker = null;

    Boolean isMakerClicked = false;

    Marker AcommoMarker = null;
    Marker AcommoMarker2 = null;
    Marker AcommoMarker3 = null;

    Button button = null;
    TextView textView = null;

    String title = " ";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_map);

        FragmentManager fragmentManager = getFragmentManager();
        MapFragment mapFragment = (MapFragment)fragmentManager
                .findFragmentById(R.id.map);
        mapFragment.getMapAsync(this);

        intent = getIntent();
        activityCode = intent.getIntExtra("index", 1);

        button = (Button)findViewById(R.id.button10);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                //button.setText("!!!!!!");

                if(isMakerClicked){
                    if(activityCode == 5){

                    }else if(activityCode == 4){

                    }else if(activityCode == 3){
                        Uri uri = Uri.parse("https://www.bustrain.net/%ED%8F%89%EC%B0%BD%EB%B2%84%EC%8A%A4%ED%84%B0%EB%AF%B8%EB%84%90-%EC%8B%9C%EA%B0%84%ED%91%9C%EC%8B%9C%EC%99%B8%EB%B2%84%EC%8A%A4%ED%84%B0%EB%AF%B8%EB%84%90-%EC%8B%9C%EA%B0%84%ED%91%9C/");
                        Intent it  = new Intent(Intent.ACTION_VIEW,uri);
                        startActivity(it);
                    }else if(activityCode == 17){
                        if(title.equals("Phoenix Pyeongchang Hotel")){
                            Uri uri = Uri.parse("http://phoenixhnr.co.kr/pyeongchang/index");
                            Intent it  = new Intent(Intent.ACTION_VIEW,uri);
                            startActivity(it);
                        }else if(title.equals("Hotel Atrium Pyeongchang")){
                            Uri uri = Uri.parse("http://www.atriumthewhite.com/");
                            Intent it  = new Intent(Intent.ACTION_VIEW,uri);
                            startActivity(it);
                        }else if(title.equals("Dragon Valley Hotel")){
                            Uri uri = Uri.parse("http://korean.visitkorea.or.kr/kor/bz15/where/where_tour.jsp?cid=142814");
                            Intent it  = new Intent(Intent.ACTION_VIEW,uri);
                            startActivity(it);
                        }
                    }else if(activityCode == 18){

                    }else if(activityCode == 19){
                        if(title.equals("House BongBong")){
                            Uri uri = Uri.parse("http://www.housebonbon.com/");
                            Intent it  = new Intent(Intent.ACTION_VIEW,uri);
                            startActivity(it);
                        }else if(title.equals("Guest Roomatiel")){
                            Uri uri = Uri.parse("http://kkalover1.blog.me/70035710087");
                            Intent it  = new Intent(Intent.ACTION_VIEW,uri);
                            startActivity(it);
                        }
                    }else if(activityCode == 20){
                        if(title.equals("Yongpyong Resort Tower Condominium")){
                            Uri uri = Uri.parse("http://www.yongpyong.co.kr/");
                            Intent it  = new Intent(Intent.ACTION_VIEW,uri);
                            startActivity(it);
                        }else if(title.equals("Green Hills Resort")){
                            Uri uri = Uri.parse("https://www.expedia.co.kr/Phan-Thiet-Hotels-Green-Hill-Resort-Spa.h6094124.Hotel-Information");
                            Intent it  = new Intent(Intent.ACTION_VIEW,uri);
                            startActivity(it);
                        }else if(title.equals("Elf Spa Resort")){
                            Uri uri = Uri.parse("http://www.elfpension.com/");
                            Intent it  = new Intent(Intent.ACTION_VIEW,uri);
                            startActivity(it);
                        }
                    }else if(activityCode == 21) {
                        if (title.equals("Yangyang International Airport")) {
                            Uri uri = Uri.parse("http://www.airport.co.kr/yangyang/main.do");
                            Intent it = new Intent(Intent.ACTION_VIEW, uri);
                            startActivity(it);
                        } else if (title.equals("Incheon International Airport (ICN)")) {
                            Uri uri = Uri.parse("http://www.airport.kr/pa/ko/d/index.jsp");
                            Intent it = new Intent(Intent.ACTION_VIEW, uri);
                            startActivity(it);
                        }
                    }else if(activityCode == 11){
                        Uri uri = Uri.parse("tel:033-332-4000");
                        Intent it = new Intent(Intent.ACTION_DIAL, uri);
                        startActivity(it);
                    }else if(activityCode == 12){
                        if (title.equals("Uri Hospital")) {
                            Uri uri = Uri.parse("tel:033-335-8275");
                            Intent it = new Intent(Intent.ACTION_DIAL, uri);
                            startActivity(it);
                        } else if (title.equals("Sungsim Hospital")) {
                            Uri uri = Uri.parse("tel:033-334-1934");
                            Intent it = new Intent(Intent.ACTION_DIAL, uri);
                            startActivity(it);
                        }
                    }else if(activityCode == 16){
                        Uri uri = Uri.parse("tel:033-335-3151");
                        Intent it = new Intent(Intent.ACTION_DIAL, uri);
                        startActivity(it);
                    }
                }

            }
        });

        textView = (TextView)findViewById(R.id.textView4);
        textView.setText("Code:" + activityCode);

    }

    @Override
    public void onMapReady(final GoogleMap map) {

        map.setOnMarkerClickListener(
                new GoogleMap.OnMarkerClickListener() {
                    @Override
                    public boolean onMarkerClick(Marker marker) {

                        //button.setText("!!!!!!");
                        isMakerClicked = true;
                        title = marker.getSnippet();
                        textView.setText("Code:" + activityCode + " Clicked");
                        return false;
                    }
                }
        );

        if(activityCode == 2){

        }else if(activityCode == 4){
            LatLng SEOUL = new LatLng(37.3694945, 128.3910656);

            MarkerOptions markerOptions = new MarkerOptions();
            markerOptions.position(SEOUL);
            markerOptions.title("평창 시내 버스 정류장(Pyeong-Chang Local Bus Station)");
            markerOptions.snippet("Local Bus");
            map.addMarker(markerOptions);
            BusStationMarker = map.addMarker(markerOptions);

            map.moveCamera(CameraUpdateFactory.newLatLng(SEOUL));
            map.animateCamera(CameraUpdateFactory.zoomTo(10));
        }else if(activityCode == 3){
            LatLng SEOUL = new LatLng(37.3664093, 128.3923528);

            MarkerOptions markerOptions = new MarkerOptions();
            markerOptions = new MarkerOptions();
            markerOptions.position(SEOUL);
            markerOptions.title("평창 버스 터미널(Pyeong-Chang Bus terminal)");
            markerOptions.snippet("Bus Express");
            BusTerminalMarker = map.addMarker(markerOptions);

            map.moveCamera(CameraUpdateFactory.newLatLng(SEOUL));
            map.animateCamera(CameraUpdateFactory.zoomTo(10));
        }else if(activityCode == 17){
            LatLng SEOUL = new LatLng(37.6322016, 128.3600107);

            MarkerOptions markerOptions = new MarkerOptions();
            markerOptions = new MarkerOptions();
            markerOptions.position(SEOUL);
            markerOptions.title("평창 호텔(Pyeong-Chang Hotel)");
            markerOptions.snippet("Phoenix Pyeongchang Hotel");
            AcommoMarker = map.addMarker(markerOptions);

            map.moveCamera(CameraUpdateFactory.newLatLng(SEOUL));
            map.animateCamera(CameraUpdateFactory.zoomTo(10));

            LatLng SEOUL2 = new LatLng(37.6376393, 128.3600107);

            MarkerOptions markerOptions2 = new MarkerOptions();
            markerOptions2 = new MarkerOptions();
            markerOptions2.position(SEOUL2);
            markerOptions2.title("평창 호텔(Pyeong-Chang Hotel)");
            markerOptions2.snippet("Dragon Valley Hotel");
            AcommoMarker2 = map.addMarker(markerOptions);

            LatLng SEOUL3 = new LatLng(37.6044633, 128.3466212);

            MarkerOptions markerOptions3 = new MarkerOptions();
            markerOptions3 = new MarkerOptions();
            markerOptions3.position(SEOUL3);
            markerOptions3.title("평창 호텔(Pyeong-Chang Hotel)");
            markerOptions3.snippet("Hotel Atrium Pyeongchang");
            AcommoMarker3 = map.addMarker(markerOptions);

        }else if(activityCode == 18){
            LatLng SEOUL = new LatLng(37.662346, 128.5347614);

            MarkerOptions markerOptions = new MarkerOptions();
            markerOptions = new MarkerOptions();
            markerOptions.position(SEOUL);
            markerOptions.title("평창 모텔(Pyeong-Chang Motel)");
            markerOptions.snippet("Bus Express");
            BusTerminalMarker = map.addMarker(markerOptions);

            map.moveCamera(CameraUpdateFactory.newLatLng(SEOUL));
            map.animateCamera(CameraUpdateFactory.zoomTo(10));
        }else if(activityCode == 19){
            LatLng SEOUL = new LatLng(37.7076998, 128.6865007);

            MarkerOptions markerOptions = new MarkerOptions();
            markerOptions = new MarkerOptions();
            markerOptions.position(SEOUL);
            markerOptions.title("평창 게스트하우스(Pyeong-Chang Guesthouse)");
            markerOptions.snippet("House BongBong");
            AcommoMarker = map.addMarker(markerOptions);

            map.moveCamera(CameraUpdateFactory.newLatLng(SEOUL));
            map.animateCamera(CameraUpdateFactory.zoomTo(10));

            LatLng SEOUL2 = new LatLng(37.6376393, 128.3600107);

            MarkerOptions markerOptions2 = new MarkerOptions();
            markerOptions2 = new MarkerOptions();
            markerOptions2.position(SEOUL2);
            markerOptions2.title("평창 게스트하우스(Pyeong-Chang Guesthouse)");
            markerOptions2.snippet("Guest Roomatiel");
            AcommoMarker2 = map.addMarker(markerOptions2);

            LatLng SEOUL3 = new LatLng(37.6044633, 128.3466212);

            MarkerOptions markerOptions3 = new MarkerOptions();
            markerOptions3 = new MarkerOptions();
            markerOptions3.position(SEOUL3);
            markerOptions3.title("평창 게스트하우스(Pyeong-Chang Guesthouse)");
            markerOptions3.snippet("Hotel Atrium Pyeongchang");
            AcommoMarker3 = map.addMarker(markerOptions3);
        }else if(activityCode == 20){
            LatLng SEOUL = new LatLng(37.6475395, 128.6695576);

            MarkerOptions markerOptions = new MarkerOptions();
            markerOptions = new MarkerOptions();
            markerOptions.position(SEOUL);
            markerOptions.title("평창 리조트 (Pyeong-Chang Resort)");
            markerOptions.snippet("Yongpyong Resort Tower Condominium");
            AcommoMarker = map.addMarker(markerOptions);

            map.moveCamera(CameraUpdateFactory.newLatLng(SEOUL));
            map.animateCamera(CameraUpdateFactory.zoomTo(10));

            LatLng SEOUL2 = new LatLng(37.6376393, 128.3600107);

            MarkerOptions markerOptions2 = new MarkerOptions();
            markerOptions2 = new MarkerOptions();
            markerOptions2.position(SEOUL2);
            markerOptions2.title("평창 리조트(Pyeong-Chang Resort)");
            markerOptions2.snippet("Green Hills Resort");
            AcommoMarker2 = map.addMarker(markerOptions);

            LatLng SEOUL3 = new LatLng(37.6044633, 128.3466212);

            MarkerOptions markerOptions3 = new MarkerOptions();
            markerOptions3 = new MarkerOptions();
            markerOptions3.position(SEOUL3);
            markerOptions3.title("평창 리조트(Pyeong-Chang Resort)");
            markerOptions3.snippet("Elf Spa Resort");
            AcommoMarker3 = map.addMarker(markerOptions);
        }else if(activityCode == 21){
            LatLng SEOUL = new LatLng(37.8282011, 127.9371766);

            MarkerOptions markerOptions = new MarkerOptions();
            markerOptions = new MarkerOptions();
            markerOptions.position(SEOUL);
            markerOptions.title("평창 공항 (Pyeong-Chang Airport)");
            markerOptions.snippet("Yangyang International Airport ");
            AcommoMarker = map.addMarker(markerOptions);

            map.moveCamera(CameraUpdateFactory.newLatLng(SEOUL));
            map.animateCamera(CameraUpdateFactory.zoomTo(10));

            LatLng SEOUL2 = new LatLng(37.474288, 126.6050843);

            MarkerOptions markerOptions2 = new MarkerOptions();
            markerOptions2 = new MarkerOptions();
            markerOptions2.position(SEOUL2);
            markerOptions2.title("평창 공항(Pyeong-Chang Emgergency)");
            markerOptions2.snippet("Incheon International Airport (ICN)");
            AcommoMarker2 = map.addMarker(markerOptions);
        }else if(activityCode == 11 || activityCode == 10){
            LatLng SEOUL = new LatLng(37.3902447, 128.3506414);

            MarkerOptions markerOptions = new MarkerOptions();
            markerOptions = new MarkerOptions();
            markerOptions.position(SEOUL);
            markerOptions.title("평창 응급실 (Pyeong-Chang Emgergency)");
            markerOptions.snippet("Pyeong-Chang Medical Center");
            AcommoMarker = map.addMarker(markerOptions);

            map.moveCamera(CameraUpdateFactory.newLatLng(SEOUL));
            map.animateCamera(CameraUpdateFactory.zoomTo(10));
        }else if(activityCode == 12){
            LatLng SEOUL = new LatLng(37.5842301, 128.4593573);

            MarkerOptions markerOptions = new MarkerOptions();
            markerOptions = new MarkerOptions();
            markerOptions.position(SEOUL);
            markerOptions.title("평창 병원 (Pyeong-Chang hospital)");
            markerOptions.snippet("Sungsim Hospital");
            AcommoMarker = map.addMarker(markerOptions);

            map.moveCamera(CameraUpdateFactory.newLatLng(SEOUL));
            map.animateCamera(CameraUpdateFactory.zoomTo(10));

            LatLng SEOUL2 = new LatLng(37.6156195, 128.3833507);

            MarkerOptions markerOptions2 = new MarkerOptions();
            markerOptions2 = new MarkerOptions();
            markerOptions2.position(SEOUL2);
            markerOptions2.title("평창 병원(Pyeong-Chang hospital)");
            markerOptions2.snippet("Uri Hospital");
            AcommoMarker2 = map.addMarker(markerOptions2);
        }else if(activityCode == 16){
            LatLng SEOUL = new LatLng(37.615427, 128.3782362);

            MarkerOptions markerOptions = new MarkerOptions();
            markerOptions = new MarkerOptions();
            markerOptions.position(SEOUL);
            markerOptions.title("평창 약국 (Pyeong-Chang Pharmacy)");
            markerOptions.snippet("Bopyeong Pharmacy");
            AcommoMarker = map.addMarker(markerOptions);

            map.moveCamera(CameraUpdateFactory.newLatLng(SEOUL));
            map.animateCamera(CameraUpdateFactory.zoomTo(10));
        }


    }

}
