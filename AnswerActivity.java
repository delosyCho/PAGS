package com.example.administrator.opencvtest;

import android.content.Intent;
import android.net.Uri;
import android.support.annotation.DrawableRes;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.Layout;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

public class AnswerActivity extends AppCompatActivity {

    int answerCode = 0;
    int currentIndex = 0;
    int MaxNum = 1;

    Button button = null;
    ImageView imageview = null;
    TextView textView = null;
    Spinner spinner;
    ImageButton nextButton;

    int[] resourses = new int[10];
    String[] tags = new String[10];

    Uri[] Uris = new Uri[10];
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_answer);

        Intent intent = getIntent();
        intent = getIntent();
        answerCode = intent.getIntExtra("index", 1);

        textView = (TextView)findViewById(R.id.textView7);

        nextButton = (ImageButton) findViewById(R.id.imageButton3);
        nextButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentIndex += 1;

                if(MaxNum == currentIndex){
                    currentIndex = 0;
                }

                imageview.setImageResource(resourses[currentIndex]);
                textView.setText(tags[currentIndex]);
            }
        });

        button = (Button)findViewById(R.id.button6);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent it = new Intent(Intent.ACTION_VIEW, Uris[currentIndex]);
                startActivity(it);
            }
        });

        if(answerCode == 8){
            MaxNum = 5;
            resourses[0] = R.drawable.food1;
            resourses[1] = R.drawable.food2;
            resourses[2] = R.drawable.food3;
            resourses[3] = R.drawable.food4;
            resourses[4] = R.drawable.food5;

            tags[0] = "여행에서 빠질 수 없는 즐거움 중 하나가 바로 ‘음식’입니다. 새로운 맛을 찾았다면 이미 그 여행은 성공한 것이나 다름없는데요. 오늘은 여행을 준비하는 분들이 꼭 한 번 들러야 할 전국의 유명 빵집을 소개하려 합니다. 출출할 때 간편하게 즐길 수 있고, 새로운 맛으로 여행을 더 특별하게 만들어주는 빵. 지금부터 전국 빵집 명소에서 맛있는 휴식을 마음껏 즐겨보시기 바랍니다. 빵을 찾아 떠나는 여행, 지금부터 시작할게요. ";
            Uris[0] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/korea/all/view?menuId=316&bbsId=31&langSeCd=ko&rows=11&cnId=1738&pageNo=1&statusCd=");

            tags[1] = "힘이 불끈~ 추위는 안녕~ 맛과 영양으로 가득 채운 겨울철 보양식을 맛볼 준비되셨다면 지금 바로 바다로~ 겨울을 맞아 풍성해진 바다의 보양식을 소개합니다!";
            Uris[1] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/korea/all/view?menuId=316&bbsId=31&langSeCd=ko&rows=25&cnId=1700&pageNo=3&statusCd=");

            tags[2] = "쌀쌀하게 불어오는 바람 사이로 풍기는 그리웠던 냄새들! 기온이 내려간 만큼 거리를 걷다 만나게 되는 따뜻한 유혹을 뿌리치기가 너무 어려워지죠. 쌀쌀해진 계절에 생각나는 길거리 음식 Best of Best 결승전! 여러분이 꼽은 가장 맛있어 보이는 길거리 음식은 무엇인가요?";
            Uris[2] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/korea/all/view?menuId=316&bbsId=31&langSeCd=ko&rows=28&cnId=1690&pageNo=3&statusCd=");

            tags[3] = "고추장 양념에 재워둔 닭고기와 갖은 채소를 둥그렇고 커다란 무쇠 프라이팬에 넣고 매콤하게 구워 먹는 춘천닭갈비.\n" +
                    "\n";
            Uris[3] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/korea/all/view?menuId=316&bbsId=31&langSeCd=ko&rows=33&cnId=1672&pageNo=3&statusCd=");

            tags[4] = "우리나라 사람들이 가장 즐겨 먹는 음식으로 겨울에는 불고기가, 여름에는 냉면이꼽힌다.\n" +
                    "물냉면은 크게 평양식과 함흥식으로 나뉜다. 평양식 냉면은 메밀이 많이 함유되어 국수에 힘이 없고 툭툭 끊어지며 국물이 맑고 담백한 것이 특징이다. 함흥식냉면은 감자 전분이나 고구마 전분의 함량이 많아 국수가 질기므로 오독오독 씹는재미가 있고 육수에 식초와 겨자를 많이 넣어 먹어야 제 맛이 난다.";
            Uris[4] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/korea/all/view?menuId=316&bbsId=31&langSeCd=ko&rows=36&cnId=1666&pageNo=3&statusCd=");
        }else if(answerCode == 6){
            MaxNum = 3;
            resourses[0] = R.drawable.accomm1;
            resourses[1] = R.drawable.accomm2;
            resourses[2] = R.drawable.accomm3;

            tags[0] = "평창 숙박";
            Uris[0] = Uri.parse("http://korean.visitkorea.or.kr/kor/bz15/where/where_main_search.jsp?type=B&areaCode=32&sigunguCode=15&catAll1=all&catAll2=all&catAll3=all");

            tags[1] = "강릉 숙박";
            Uris[1] = Uri.parse("http://korean.visitkorea.or.kr/kor/bz15/where/where_main_search.jsp?type=B&areaCode=32&sigunguCode=1&catAll1=all&catAll2=all&catAll3=all");

            tags[2] = "정선 숙박";
            Uris[2] = Uri.parse("http://korean.visitkorea.or.kr/kor/bz15/where/where_main_search.jsp?type=B&areaCode=32&sigunguCode=11&catAll1=all&catAll2=all&catAll3=all");
        }else if(answerCode == 5){
            MaxNum = 5;
            resourses[0] = R.drawable.transport1;
            resourses[1] = R.drawable.transport2;
            resourses[2] = R.drawable.transport3;
            resourses[3] = R.drawable.transport4;
            resourses[4] = R.drawable.transport5;


            tags[0] = "관중 셔틀버스(Free tourist bus service)";
            Uris[0] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/transportation/staticcontents?menuId=912");

            tags[1] = "KTX(Fast train)";
            Uris[1] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/transportation/staticcontents?menuId=912");

            tags[2] = "비행기(Airplane)";
            Uris[2] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/transportation/staticcontents?menuId=912");

            tags[3] = "시외 고속 버스(Express Bus)";
            Uris[3] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/transportation/staticcontents?menuId=912");

            tags[4] = "고속도로(자동차) : Highway(Car)";
            Uris[4] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/transportation/staticcontents?menuId=912");

        }else if(answerCode == 13){
            MaxNum = 5;
            resourses[0] = R.drawable.play1;
            resourses[1] = R.drawable.play2;
            resourses[2] = R.drawable.play3;
            resourses[3] = R.drawable.play4;
            resourses[4] = R.drawable.play5;

            tags[0] = "이번에 소개할 축제는 바로 유네스코가 지정한 ‘인류구전 및 무형유산 걸작’인 '강릉 단오제'입니다. 올해의 강릉 단오제는 5월 27일부터 6월 3일까지 진행되는데요. 춤, 노래, 연극, 제사, 굿 등 우리나라 전통 무형 유산의 모든 장르를 만날 수 있는 종합 예술의 장";
            Uris[0] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/city/all/view?menuId=312&bbsId=30&langSeCd=ko&rows=2&cnId=1770&pageNo=1&statusCd=");

            tags[1] = "수없이 사라져가는 전통문화들 사이에서 꿋꿋하게 옛 원형 그대로 전통을 이어가고 있는 강릉 단오제! 음력 5월 5일인 단오는 한 해의 기운이 가장 활발해지는 날로 농경사회였던 예부터 24절기 중 가장 신성하고 중요한 날로 여겨졌어요. 단오제는 지방의 주민들, 관속, 무당이 모두 화합하여 참여하는 중요한 제례의식";
            Uris[1] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/city/all/view?menuId=312&bbsId=30&langSeCd=ko&rows=33&cnId=1678&pageNo=3&statusCd=");

            tags[2] = "평창 패러글라이딩 체험";
            Uris[2] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/city/all/view?menuId=312&bbsId=30&langSeCd=ko&rows=34&cnId=1675&pageNo=3&statusCd=");

            tags[3] = "재미있는 주제로 강원도의 다양한 가을 풍경을 즐길 수 있는 대표 축제들";
            Uris[3] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/city/all/view?menuId=312&bbsId=30&langSeCd=ko&rows=38&cnId=1648&pageNo=4&statusCd=");

            tags[4] = "평창군 진부면에서는 2014년 12월 20일부터 2015년 2월 2일까지 <평창송어축제>를 개최한다. <평창송어축제>에서는 선조들의 삶을 축제로 승화시켜 눈과 얼음, 송어가 함께하는 겨울이야기라는 주제로 매년 겨울마다 송어축제의 장이 펼쳐진다. 송어낚시와 썰매체험 등 다양한 체험 프로그램과 함께 진정한 겨울 축제의 즐거움을 만끽할 수 있다. 송어는 연어과에 속하는 소하형 어종으로 한국의 동해와 동해로 흐르는 일부 하천에 분포하며, 북한, 일본, 연해주 등지에도 분포한다. 특히 평창군은 국내 최대의 송어 양식지이며, 평창의 맑은 물에서 자란 송어는 부드럽고 쫄깃쫄깃한 식감이 일품이다.";
            Uris[4] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/city/all/view?menuId=312&bbsId=30&langSeCd=ko&rows=47&cnId=1638&pageNo=4&statusCd=");

        }else if(answerCode == 15){
            MaxNum = 3;
            resourses[0] = R.drawable.tour1;
            resourses[1] = R.drawable.tour2;
            resourses[2] = R.drawable.tour3;

            tags[0] = "올림픽 플라자";
            Uris[0] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/city/all/view?menuId=312&bbsId=30&langSeCd=ko&rows=2&cnId=1770&pageNo=1&statusCd=");

            tags[1] = "올림픽 스타디움";
            Uris[1] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/city/all/view?menuId=312&bbsId=30&langSeCd=ko&rows=33&cnId=1678&pageNo=3&statusCd=");

            tags[2] = "성화대";
            Uris[2] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/culture/city/all/view?menuId=312&bbsId=30&langSeCd=ko&rows=34&cnId=1675&pageNo=3&statusCd=");
        }else if(answerCode == 14){
            MaxNum = 3;
            resourses[0] = R.drawable.city1;
            resourses[1] = R.drawable.city2;
            resourses[2] = R.drawable.city3;

            tags[0] = "평창";
            Uris[0] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/information/pyeongchang/staticcontents?menuId=829");

            tags[1] = "강릉";
            Uris[1] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/information/gangneung/staticcontents?menuId=830");

            tags[2] = "정선";
            Uris[2] = Uri.parse("https://www.pyeongchang2018.com/ko/spectator/information/jeongseon/staticcontents?menuId=831");
        }else if(answerCode == 23){
            MaxNum = 1;
            resourses[0] = R.drawable.market1;


            tags[0] = "평창 올림픽 시장";
            Uris[0] = Uri.parse("http://terms.naver.com/entry.nhn?docId=3347438&cid=51381&categoryId=51381");
        }

        imageview = (ImageView)findViewById(R.id.imageView3);
        imageview.setImageResource(resourses[currentIndex]);
        textView.setText(tags[currentIndex]);
    }
}
