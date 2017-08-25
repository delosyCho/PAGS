package com.example.administrator.opencvtest;

/**
 * Created by Administrator on 2017-05-21.
 */

public class WordBag {
    public String[] Words;
    public double[] Scores;
    int NumberOfWord;
    public WordBag(int numberOfWord){
        NumberOfWord = numberOfWord;

        Words = new String[NumberOfWord];
        Scores = new double[NumberOfWord];
    }

    public WordBag(String Word){
        String[] Toekns = Word.split("@");
        int Length = Toekns.length;
        NumberOfWord = Length - 2;

        Words = new String[Toekns.length - 1];
        Scores = new double[Toekns.length - 1];

        for(int i = 1; i < Length - 1; i++){
            Scores[i - 1] = Double.parseDouble(Toekns[i].split("#")[1]);
            Words[i - 1] = Toekns[i].split("#")[0];
        }

    }

    public double getScore(String str){
        double score = 0;
        String[] toekns = str.split(" ");
        for(int i = 0; i < NumberOfWord; i++){

            for(int j = 0; j < toekns.length; j++){
                if(toekns[j].compareTo(Words[i]) == 0){
                    score += Scores[i];
                }
            }

        }

        return score;
    }
}
