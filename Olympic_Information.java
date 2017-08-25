package com.example.administrator.opencvtest;

import android.content.Intent;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import org.w3c.dom.Text;

public class Olympic_Information extends AppCompatActivity {

    ImageButton imageButton = null;
    ImageView imageView = null;

    String[] Urls = new String[5];

    Button button = null;
    TextView textView = null;

    int Page = 0;
    String[] Description = new String[5];
    int[] Resources = {R.drawable.olympic2, R.drawable.olympic3, R.drawable.olympic1
    , R.drawable.olympic4, R.drawable.olympic5};

    String[] texts = new String[5];

    public void setResource(){
        texts[0] = "동계 올림픽의 역사 - 자세히 보기";
        texts[1] = "한국 동계 올림픽 역사 - 자세히 보기";
        texts[2] = "평창 동계 올림픽 - 자세히 보기";
        texts[3] = "평창 동계 올림픽 효과  - 자세히 보기";
        texts[4] = "평창 올림픽 마스코트 - 자세히 보기";

        Urls[0] = "https://www.pyeongchang2018.com/ko/olympicstory/history/global/staticcontents?menuId=245";
        Urls[1] = "https://www.pyeongchang2018.com/ko/olympicstory/history/global/staticcontents?menuId=245";
        Urls[2] = "https://www.pyeongchang2018.com/ko/pyeongchang2018/olympics/introduction/staticcontents?menuId=203";
        Urls[3] = "https://www.pyeongchang2018.com/ko/pyeongchang2018/olympics/effects/staticcontents?menuId=222";
        Urls[4] = "https://www.pyeongchang2018.com/ko/pyeongchang2018/olympics/mascot/index";


        Description[0] = "1924 \n"+
                "\n"+
                "제1회 프랑스 샤모니\n"+
                "01.25 ~ 02.05\n"+
                "참가국 : 16 ㅣ 세부종목 : 16 ㅣ 참가선수 : 258\n"+
                "\n"+
                "1928 \n"+
                "\n"+
                "제2회 스위스 생모리츠\n"+
                "02.11 ~ 02.19\n"+
                "참가국 : 25 ㅣ 세부종목 : 14 ㅣ 참가선수 : 464\n"+
                "\n"+
                "1932 \n"+
                "\n"+
                "제3회 미국 레이크플래시드\n"+
                "02.04 ~ 02.15\n"+
                "참가국 : 17 ㅣ 세부종목 : 14 ㅣ 참가선수 : 252\n"+
                "\n"+
                "1936 \n"+
                "\n"+
                "제4회 독일 가르미쉬파르텐키르헨\n"+
                "02.06 ~ 02.16\n"+
                "참가국 : 28 ㅣ 세부종목 : 17 ㅣ 참가선수 : 646\n"+
                "\n"+
                "1948 \n"+
                "\n"+
                "제5회 스위스 생모리츠\n"+
                "01.30 ~ 02.08\n"+
                "참가국 : 28 ㅣ 세부종목 : 22 ㅣ 참가선수 : 669\n"+
                "\n"+
                "1952 \n"+
                "\n"+
                "제6회 노르웨이 오슬로\n"+
                "02.14 ~ 02.25\n"+
                "참가국 : 30 ㅣ 세부종목 : 22 ㅣ 참가선수 : 694\n"+
                "\n"+
                "1956 \n"+
                "\n"+
                "제7회 이탈리아 코르티나담페초\n"+
                "01.26 ~ 02.05\n"+
                "참가국 : 32 ㅣ 세부종목 : 24 ㅣ 참가선수 : 821\n"+
                "\n"+
                "1960 \n"+
                "\n"+
                "제8회 미국 스퀘벨리\n"+
                "02.18 ~ 02.28\n"+
                "참가국 : 30 ㅣ 세부종목 : 27 ㅣ 참가선수 : 665\n"+
                "\n"+
                "1964 \n"+
                "\n"+
                "제9회 오스트리아 인스부르크\n"+
                "01.29 ~ 02.09\n"+
                "참가국 : 36 ㅣ 세부종목 : 34 ㅣ 참가선수 : 1,091\n"+
                "\n"+
                "1968 \n"+
                "\n"+
                "제10회 프랑스 그레노블\n"+
                "02.06 ~ 02.18\n"+
                "참가국 : 36 ㅣ 세부종목 : 35 ㅣ 참가선수 : 1,158\n"+
                "\n"+
                "1972 \n"+
                "\n"+
                "제11회 일본 샷포로\n"+
                "02.03 ~ 02.13\n"+
                "참가국 : 35 ㅣ 세부종목 : 35 ㅣ 참가선수 : 1,006\n"+
                "\n"+
                "1976 \n"+
                "\n"+
                "제12회 오스트리아 인스부르크\n"+
                "02.04 ~ 02.15\n"+
                "참가국 : 37 ㅣ 세부종목 : 22 ㅣ 참가선수 : 669\n"+
                "\n"+
                "1980 \n"+
                "\n"+
                "제13회 미국 레이크플레시드\n"+
                "02.13 ~ 02.24\n"+
                "참가국 : 37 ㅣ 세부종목 : 38 ㅣ 참가선수 : 1,072\n"+
                "\n"+
                "1984 \n"+
                "\n"+
                "제14회 유고슬라비아 사라예보\n"+
                "02.08 ~ 02.19\n"+
                "참가국 : 49 ㅣ 세부종목 : 39 ㅣ 참가선수 : 1,272\n"+
                "\n"+
                "1988 \n"+
                "\n"+
                "제15회 캐나다 캘거리\n"+
                "02.13 ~ 02.28\n"+
                "참가국 : 57 ㅣ 세부종목 : 46 ㅣ 참가선수 : 1,423\n"+
                "\n"+
                "1992 \n"+
                "\n"+
                "제16회 프랑스 알베르빌\n"+
                "02.08 ~ 02.23\n"+
                "참가국 : 64 ㅣ 세부종목 : 57 ㅣ 참가선수 : 1,801\n"+
                "\n"+
                "1994 \n"+
                "\n"+
                "제17회 노르웨이 릴레함메르\n"+
                "02.12 ~ 02.27\n"+
                "참가국 : 67 ㅣ 세부종목 : 61 ㅣ 참가선수 : 1,737\n"+
                "\n"+
                "1998 \n"+
                "\n"+
                "제18회 일본 나가노\n"+
                "02.07 ~ 02.22\n"+
                "참가국 : 72 ㅣ 세부종목 : 68 ㅣ 참가선수 : 2,176\n"+
                "\n"+
                "2002 \n"+
                "\n"+
                "제19회 미국 솔트레이크시티\n"+
                "02.08 ~ 02.24\n"+
                "참가국 : 77 ㅣ 세부종목 : 78 ㅣ 참가선수 : 2,399\n"+
                "\n"+
                "2006 \n"+
                "\n"+
                "제20회 이탈리아 토리노\n"+
                "02.10 ~ 02.26\n"+
                "참가국 : 80 ㅣ 세부종목 : 84 ㅣ 참가선수 : 2,508\n"+
                "\n"+
                "2010 \n"+
                "\n"+
                "제21회 캐나다 밴쿠버\n"+
                "02.12 ~ 02.28\n"+
                "참가국 : 82 ㅣ 세부종목 : 86 ㅣ 참가선수 : 2,566\n"+
                "\n"+
                "2014 \n"+
                "\n"+
                "제22회 러시아 소치\n"+
                "02.07 ~ 02.23\n"+
                "참가국 : 88 ㅣ 세부종목 : 98 ㅣ 참가선수 : 2,780\n"+
                "\n"+
                "2018 \n"+
                "\n"+
                "제23회 대한민국 평창 (예정)\n";


        Description[1] = "1948 \n" +
                "\n" +
                "제5회 스위스 생모리츠\n" +
                "01.30 ~ 02.08\n" +
                "참가국 : 28 ㅣ 세부종목 : 22 ㅣ 참가선수 : 669\n" +
                "\n" +
                "1952 \n" +
                "\n" +
                "제6회 노르웨이 오슬로\n" +
                "02.14 ~ 02.25\n" +
                "참가국 : 30 ㅣ 세부종목 : 22 ㅣ 참가선수 : 694\n" +
                "※ 대한민국은 불참\n" +
                "\n" +
                "1956 \n" +
                "\n" +
                "제7회 이탈리아 코르티나담페초\n" +
                "01.26 ~ 02.05\n" +
                "참가국 : 32 ㅣ 세부종목 : 24 ㅣ 참가선수 : 821\n" +
                "\n" +
                "1960 \n" +
                "\n" +
                "제8회 미국 스퀘벨리\n" +
                "02.18 ~ 02.28\n" +
                "참가국 : 30 ㅣ 세부종목 : 27 ㅣ 참가선수 : 665\n" +
                "\n" +
                "1964 \n" +
                "\n" +
                "제9회 오스트리아 인스부르크\n" +
                "01.29 ~ 02.09\n" +
                "참가국 : 36 ㅣ 세부종목 : 34 ㅣ 참가선수 : 1,091\n" +
                "\n" +
                "1968 \n" +
                "\n" +
                "제10회 프랑스 그레노블\n" +
                "02.06 ~ 02.18\n" +
                "참가국 : 36 ㅣ 세부종목 : 35 ㅣ 참가선수 : 1,158\n" +
                "\n" +
                "1972 \n" +
                "\n" +
                "제11회 일본 샷포로\n" +
                "02.03 ~ 02.13\n" +
                "참가국 : 35 ㅣ 세부종목 : 35 ㅣ 참가선수 : 1,006\n" +
                "\n" +
                "1976 \n" +
                "\n" +
                "제12회 오스트리아 인스부르크\n" +
                "02.04 ~ 02.15\n" +
                "참가국 : 37 ㅣ 세부종목 : 22 ㅣ 참가선수 : 669\n" +
                "\n" +
                "1980 \n" +
                "\n" +
                "제13회 미국 레이크플레시드\n" +
                "02.13 ~ 02.24\n" +
                "참가국 : 37 ㅣ 세부종목 : 38 ㅣ 참가선수 : 1,072\n" +
                "\n" +
                "1984 \n" +
                "\n" +
                "제15회 캐나다 캘거리\n" +
                "02.13 ~ 02.28\n" +
                "참가국 : 57 ㅣ 세부종목 : 46 ㅣ 참가선수 : 1,423\n" +
                "\n" +
                "1988 \n" +
                "\n" +
                "제15회 캐나다 캘거리\n" +
                "02.13 ~ 02.28\n" +
                "참가국 : 57 ㅣ 세부종목 : 46 ㅣ 참가선수 : 1,423\n" +
                "\n" +
                "1992 \n" +
                "\n" +
                "제16회 프랑스 알베르빌\n" +
                "02.08 ~ 02.23\n" +
                "참가국 : 64 ㅣ 세부종목 : 57 ㅣ 참가선수 : 1,801\n" +
                "\n" +
                "순위 : 10위 (금 2, 은 1, 동 1)\n" +
                "\n" +
                "1994 \n" +
                "\n" +
                "제17회 노르웨이 릴레함메르\n" +
                "02.12 ~ 02.27\n" +
                "참가국 : 67 ㅣ 세부종목 : 61 ㅣ 참가선수 : 1,737\n" +
                "\n" +
                "순위 : 6위 (금 4, 은 1, 동 1)\n" +
                "\n" +
                "1998 \n" +
                "\n" +
                "제18회 일본 나가노\n" +
                "02.07 ~ 02.22\n" +
                "참가국 : 72 ㅣ 세부종목 : 68 ㅣ 참가선수 : 2,176\n" +
                "\n" +
                "순위 : 9위 (금 3, 은 1, 동 2)\n" +
                "\n" +
                "" +
                "2002 \n" +
                "\n" +
                "제19회 미국 솔트레이크시티\n" +
                "02.08 ~ 02.24\n" +
                "참가국 : 77 ㅣ 세부종목 : 78 ㅣ 참가선수 : 2,399\n" +
                "\n" +
                "순위 : 14위 (금 2, 은 2)\n" +
                "\n" +
                "" +
                "2006 \n" +
                "\n" +
                "제20회 이탈리아 토리노\n" +
                "02.08 ~ 02.24\n" +
                "참가국 : 77 ㅣ 세부종목 : 78 ㅣ 참가선수 : 2,399\n" +
                "\n" +
                "순위 : 7위 (금 6, 은 3, 동 2)\n" +
                "\n" +
                "" +
                "\n" +
                "2010 \n" +
                "\n" +
                "제21회 캐나다 밴쿠버\n" +
                "02.12 ~ 02.28\n" +
                "참가국 : 82 ㅣ 세부종목 : 86 ㅣ 참가선수 : 2,566\n" +
                "\n" +
                "순위 : 5위 (금 6, 은 6, 동 2)\n" +
                "\n" +
                "" +
                "2014 \n" +
                "\n" +
                "제22회 러시아 소치\n" +
                "02.07 ~ 02.23\n" +
                "참가국 : 88 ㅣ 세부종목 : 98 ㅣ 참가선수 : 2,780\n" +
                "\n" +
                "순위 : 13위 (금 3, 은 3, 동 2)\n" +
                "\n" +
                "" +
                "\n";

        Description[2] = "세계인의 축제, 제23회 동계올림픽대회는 대한민국 강원도 평창에서 2018년 2월 9일부터 25일까지 17일간 개최됩니다. 대한민국 평창은 세 번의 도전 끝에 지난 2011년 7월 7일 열린 제 123차 IOC 총회에서 과반 표를 획득하며 2018년 동계올림픽 개최지로 선정되었습니다. 이로써 대한민국에서는 1988년 서울 올림픽 이후 30년 만에, 평창에서 개∙폐회식과 대부분의 설상 경기가 개최되며, 강릉에서는 빙상 종목 전 경기가, 그리고 정선에서는 알파인 스키 활강 경기가 개최될 예정입니다.";

        Description[3] = "\n" +
                "경제 활성화 기여\n" +
                "\n" +
                "\n" +
                "대회 준비단계부터 다양한 일자리 창출 등 경제활성화 기여\n" +
                " \n" +
                "\n" +
                "\n" +
                "\n" +
                "국가 브랜드 향상\n" +
                "\n" +
                "\n" +
                "국가브랜드 향상을 통해 국력과 국격을 전 세계에 떨치고 정치·경제·사회· 문화적으로 한 단계 재도약\n" +
                " \n" +
                "\n" +
                "\n" +
                "\n" +
                "지역 균형 발전\n" +
                "\n" +
                "\n" +
                "개최 지역 브랜드 가치 상승 및 사회간접자본(SOC) 확충을 통한 지역 균형 발전 도모\n" +
                " \n" +
                "\n" +
                "\n" +
                "\n" +
                "국가 발전 에너지 결집\n" +
                "\n" +
                "\n" +
                "국민화합 및 자긍심 고양을 통한 국가 발전 에너지 결집\n" +
                " \n" +
                "\n" +
                "\n" +
                "\n" +
                "첨단 산업 발전 촉진\n" +
                "\n" +
                "\n" +
                "IT·녹색산업 등 첨단산업 발전 촉진을 통해 세계시장 주도\n" +
                " \n" +
                "\n" +
                "\n" +
                "\n" +
                "선진국 진입의\n" +
                " 상징적 계기\n" +
                "\n" +
                "\n" +
                "88서울올림픽에 이은 한국올림픽의 완성으로 선진국 진입의 상징적 계기 마련\n" +
                " \n" +
                "\n" +
                "\n" +
                "\n" +
                "남북 간 화해 협력\n" +
                " 및 평화 증진\n" +
                "\n" +
                "\n" +
                "올림픽 무브먼트 실현을 통해 남북 간 화해 협력 및 평화 증진 기여\n" +
                " \n" +
                "\n" +
                "\n" +
                "\n" +
                "아시아의 동계\n" +
                " 스포츠 허브\n" +
                "\n" +
                "\n" +
                "성장 잠재력이 큰 아시아의 동계스포츠 허브로 자리매김\n" +
                "\n";

        Description[4] = "2018 평창의 마스코트";
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_olympic__information);
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

        setResource();

        textView = (TextView)findViewById(R.id.textView9);
        imageView = (ImageView)findViewById(R.id.imageView4);

        imageButton = (ImageButton)findViewById(R.id.imageButton2);
        imageButton.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){

                Page++;

                if(Page == 5){
                    Page = 0;
                }

                textView.setText(Description[Page]);
                imageView.setImageResource(Resources[Page]);
                button.setText(texts[Page]);


            }
        });

        button = (Button)findViewById(R.id.button9);
        button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){

                Intent it = new Intent(Intent.ACTION_VIEW, Uri.parse(Urls[Page]));
                startActivity(it);

            }
        });

        textView.setText(Description[Page]);
        imageView.setImageResource(Resources[Page]);
        button.setText(texts[Page]);

    }

}
