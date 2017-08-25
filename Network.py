# -*- coding : cp949 -*-
import Explolation_Attention_Model
import socket
import Improved_AoA_Reader
import numpy as np

def get_Translate(str):
    import os
    import sys
    import urllib.parse
    import urllib.request
    client_id = "HjNG5lCpBIorpjksNXKd"
    client_secret = "eauwIjFh10"
    encText = urllib.parse.quote(str)
    data = "source=ko&target=en&text=" + encText
    url = "https://openapi.naver.com/v1/language/translate"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()

    if (rescode == 200):
        response_body = response.read()
        msg = response_body.decode('utf-8')
        print(msg)

        result = msg.split('translatedText":"')[1]
        result = result.split('."}}')[0]
        result = result.split('?"}}')[0]

        print(result)

    else:
        print("Error Code:" + rescode)



paragraph = "The role of teacher is often formal and ongoing, carried out at a school or other place of formal education. In many countries, a person who wishes to become a teacher must first obtain specified professional qualifications or credentials from a university or college. These professional qualifications may include the study of pedagogy, the science of teaching. Teachers, like other professionals, may have to continue their education after they qualify, a process known as continuing professional development. Teachers may use a lesson plan to facilitate student learning, providing a course of study which is called the curriculum"
paragraph = paragraph.replace('.', '')
paragraph = paragraph.replace(',', '')
paragraph = paragraph.lower()

#print(paragraph)

qa = 'what is role of teacher'

#print(get_Translate("학교에서 선생의 역활은 무엇입니까"))

#qa_model = Explolation_Attention_Model.Seq2Seq_QA()
#a, b = qa_model.get_QA_Answer(para=paragraph, qu=qa)
#qa_model.training_continue(50)
#qa_model.training(50)
#qa_model.propagate()

#paragraph_ = paragraph.split(' ')

p_length = 125
q_length = 30

qa_input = np.zeros(shape=[q_length], dtype='<U20')
p_input = np.zeros(shape=[p_length], dtype='<U20')

TK = qa.split(' ')
for i in range(len(TK)):
    qa_input[i] = TK[i]

TK = paragraph.split(' ')
for i in range(len(TK)):
    p_input[i] = TK[i]


qa_model = Improved_AoA_Reader.Improved_AoA_Reader()
print("Network Prepare!")

HOST = ''  # 호스트를 지정하지 않으면 가능한 모든 인터페이스를 의미한다.

PORT = 17755  # 포트지정

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.bind((HOST, PORT))

s.listen(1)  # 접속이 있을때까지 기다림

conn, addr = s.accept()  # 접속 승인

print('Connected by', addr)

while True:

    data = conn.recv(1024)

    if not data: break

    qa = str(data)
    msg = str(data) + '\n'

    index = qa_model.getPropResult(paragraph_=p_input, question_=qa_input)
    result = ''
    for i in range(index, index + 5):
        result = result + paragraph[i]

    msg = result + '\n'
    print(msg)
    conn.send("".join(msg).encode(encoding='utf-8'))  # 받은 데이터를 그대로 클라이언트에 전송

conn.close()
print("Closed")