#Apache 2.0 License
#Author: Sanghyeon Cho(Pusan National Univ.) delosycho@gmail.com

import tensorflow as tf
import json
import codecs
import model
import Explolation_Attention_Model
import numpy
import os
import sys
import urllib.parse
import urllib.request
import AoA_Reader

in_path = "C:\\Users\\Administrator\\Desktop\\qadataset\\train-v1.1.json"
data = json.load(open(in_path,  'r'))

def get_Translate(str):

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
    result = ""
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

    return result


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches

if __name__ == ("__main__"):
    print('start')
    """
    data = json.load(open(in_path, 'r'))
    article = data['data'][0]
    para = article['paragraphs'][0]
    para['context'] = para['context'].replace(u'\u000A', '')
    para['context'] = para['context'].replace(u'\u00A0', ' ')
    #para['context'] = para['context'].replace('\"', '')
    # print(para['context'])
    context = para['context']
    context = "".join(context).replace('.', '')
    context = context.replace(',', '')

    paragraph = numpy.array(context.split(' '))

    wrong_loc_count = 0
    loc_diffs = []

    test = context.split(' ')
    abc = 0
    for t in test:
        abc = abc + 1
    print("Length::", abc)

    max = -999

    for article in data['data']:
        for para in article['paragraphs']:
            para['context'] = para['context'].replace(u'\u000A', '')
            para['context'] = para['context'].replace(u'\u00A0', ' ')
            # para['context'] = para['context'].replace('\"', '')

            context = para['context']
            context = context.replace('.', ' <dot>')

            length = len(context.split(' '))
            if max < length:
                max = length

    #print("Longest paragraph Length : ", length)
    #print()

    for qa in para['qas']:
        for answer in qa['answers']:
            answer['text'] = answer['text'].replace(u'\u00A0', ' ')
            text = answer['text']
            answer_start = answer['answer_start']
            if context[answer_start:answer_start + len(text)] == text:
                if text.lstrip() == text:
                    pass
                else:
                    answer_start += len(text) - len(text.lstrip())
                    answer['answer_start'] = answer_start
                    text = text.lstrip()
                    answer['text'] = text
            else:
                wrong_loc_count += 1
                text = text.lstrip()
                answer['text'] = text
                starts = list(find_all(context, text))
                if len(starts) == 1:
                    answer_start = starts[0]
                elif len(starts) > 1:
                    new_answer_start = min(starts, key=lambda s: abs(s - answer_start))
                    loc_diffs.append(abs(new_answer_start - answer_start))
                    answer_start = new_answer_start
                else:
                    raise Exception()
                answer['answer_start'] = answer_start

            answer_stop = answer_start + len(text)
            answer['answer_stop'] = answer_stop

            #print(answer_start, " : ", answer_stop, context[answer_start:answer_stop])
            context2 = list(context)
            context2[answer_start] = '#'
            context2[answer_stop - 1] = '#'
            context2 = "".join(context2)
            context2 = numpy.array(context2.split(' '))

            text = list(text)
            text[0] = '#'
            text[answer_stop - answer_start - 1] = '#'
            text = "".join(text)
            text = numpy.array(text.split(' '))

            ans_index1 = numpy.where(context2 == text[0])
            ans_index2 = numpy.where(context2 == text[text.size - 1])

            q_length = len(qa['question'].split(' '))
            question = "".join(qa['question']).replace('?', '')
            question = question.split(' ')
            tempArr_q = numpy.zeros((20), dtype="<U20")

            for i in range(q_length):
                tempArr_q[i] = question[i]

            #print(tempArr_q)

            #print("q length : ", q_length, question)
            #print(context2)
            #print(ans_index1[0][0], ans_index2[0][0], '\n', qa['question'], answer['text'])
            #print(context2[ans_index1], paragraph[ans_index1], " @ ", text[text.size - 1], context2[ans_index2], paragraph[ans_index2])

    my = list("ssssssss")
    my[0] = 'b'
    my = "".join(my)
    print("my:", my)

    input = []
    arr = [[1,1],[2,2]]

    input.append(arr)
    input.append(arr)
    input.append(arr)

    print(numpy.array(input).shape)

    string1 = "strl. str2. str3. str4. str5"
    string1 = string1.replace('.', '')
    print(string1)

    string2 = "str"

    words = []
    vectors = []

    in_path_glove = "C:\\Users\\Administrator\\Desktop\\qadataset\\glove6B200d.txt"
    glove_f = codecs.open(in_path_glove, 'r', 'utf-8')

    for line in range(1000):
        lines = glove_f.readline()
        tokens = lines.split(' ')
        words.append(tokens.pop(0))
        vectors.append(tokens)

    vectors = numpy.array((vectors), 'f').reshape((-1, 200))
    print("shape", vectors.shape)

    s = words[0]
#    print(vectors[0])

    dictionary = numpy.array(words)
    glove_arg_index = dictionary.argsort()
    dictionary.sort()

    testb = numpy.array(("asdasd", "cvxcvxc"))
    testa = numpy.zeros((10), dtype="<U20")

    testa[0] = "aaaq12qwe"
    testa[1] = "bbbasd1we"
    testa[2] = "Cccasedqwe1"
    testa[3] = "dddasdqsaeqw1"

    numberOfParagraph = 0
    numberOfQuestions = 0

    for article in data['data']:
        for para in article['paragraphs']:
            numberOfParagraph = numberOfParagraph + 1

            for qa in para['qas']:
                for answer in qa['answers']:
                    numberOfQuestions = numberOfQuestions + 1

    print(numberOfParagraph, numberOfQuestions)

    max = -999
    para = ""

    for article in data['data']:
        for para in article['paragraphs']:
            context = para['context']
            context = context.replace('.', ' .')
            context = context.replace(',', ' ,')
            context = context.replace('?', ' ?')
            context = context.replace('!', ' !')
            context = context.replace('(', ' ')
            context = context.replace(')', ' ')
            context = context.replace(u'\u2013', ' - ')
            context = context.replace(u'\u2014', ' - ')
            context = context.replace('-', ' - ')
            context = context.replace('\'', ' \' ')
            context = context.replace('\"', '')

            #context = context.replace('.', ' .')
            #context = context.replace(',', ' ,')

            length = len(context.split(' '))
            if max < length:
                max = length
                parag = context
                # print("Longest paragraph Length : ", length)
                # print()

    print(max)
    print(parag)

    parag = parag.split(' ')

    for i in range(max):
        print(parag[i], end=' ')

    wrong_count = 0
    wrong_loc_count = 0
    loc_diffs = []

    paragraph_index = 0
    question_index = 0

    numberOfParagraph = 0
    numberOfQuestions = 0

    for article in data['data']:
        for para in article['paragraphs']:
            numberOfParagraph = numberOfParagraph + 1

            for qa in para['qas']:
                for answer in qa['answers']:
                    numberOfQuestions = numberOfQuestions + 1

    quetion_maxlength = 60

    batch = []
    paragraph_arr = numpy.zeros(shape=(numberOfParagraph, 825), dtype="<U20")
    question_batch = numpy.zeros(shape=(numberOfQuestions, quetion_maxlength), dtype="<U20")
    start_index_batch = numpy.zeros(shape=(numberOfQuestions), dtype=numpy.int)
    stop_index_batch = numpy.zeros(shape=(numberOfQuestions), dtype=numpy.int)

    for article in data['data']:
        for para in article['paragraphs']:
            para['context'] = para['context'].replace(u'\u000A', '')
            para['context'] = para['context'].replace(u'\u00A0', ' ')
            # para['context'] = para['context'].replace('\"', '')

            context = para['context']
            context = context.replace('.', ' .')
            context = context.replace(',', ' ,')
            context = context.replace('?', ' ?')
            context = context.replace('!', ' !')
            context = context.replace('(', ' ')
            context = context.replace(')', ' ')
            context = context.replace(u'\u2013', ' - ')
            context = context.replace(u'\u2014', ' - ')
            context = context.replace('-', ' - ')
            context = context.replace('\'', ' \' ')
            context = context.replace('\"', '')

            print(context)
            paragraph = numpy.array(context.split(' '))
            print("Max", max, "para:", len(paragraph))

            tempArr_p = numpy.zeros((825), dtype="<U20")
            for i in range(len(paragraph)):
                tempArr_p[i] = paragraph[i]

            paragraph_arr[paragraph_index] = tempArr_p

            is_wrong = 0

            for qa in para['qas']:
                for answer in qa['answers']:
                    answer['text'] = answer['text'].replace(u'\u00A0', ' ')
                    text = answer['text']
                    answer_start = answer['answer_start']
                    if context[answer_start:answer_start + len(text)] == text:
                        if text.lstrip() == text:
                            pass
                        else:
                            answer_start += len(text) - len(text.lstrip())
                            answer['answer_start'] = answer_start
                            text = text.lstrip()
                            answer['text'] = text
                    else:
                        wrong_loc_count += 1
                        text = text.lstrip()
                        answer['text'] = text
                        starts = list(find_all(context, text))
                        if len(starts) == 1:
                            answer_start = starts[0]
                        elif len(starts) > 1:
                            new_answer_start = min(starts, key=lambda s: abs(s - answer_start))
                            loc_diffs.append(abs(new_answer_start - answer_start))
                            answer_start = new_answer_start
                        else:
                            start_index_batch[question_index] = -1
                            stop_index_batch[question_index] = -1
                            print("Raise Exception")
                            print(answer['text'], " : ", qa['question'])
                            #raise Exception()
                            is_wrong = 1
                            wrong_count = wrong_count + 1
                        answer['answer_start'] = answer_start

                    if is_wrong == 0:
                        answer_stop = answer_start + len(text)
                        answer['answer_stop'] = answer_stop

                        # print(answer_start, " : ", answer_stop, context[answer_start:answer_stop])
                        context2 = list(context)
                        context2[answer_start] = '#'
                        context2[answer_stop - 1] = '^'
                        context2 = "".join(context2)
                        context2 = numpy.array(context2.split(' '))

                        text = list(text)
                        text[0] = '#'
                        text[answer_stop - answer_start - 1] = '^'
                        text = "".join(text)
                        text = numpy.array(text.split(' '))

                        ans_index1 = numpy.where(context2 == text[0])
                        ans_index2 = numpy.where(context2 == text[text.size - 1])

                        if len(ans_index1[0]) > 0 and len(ans_index2[0]) > 0:
                            start_index_batch[question_index] = ans_index1[0][0]
                            stop_index_batch[question_index] = ans_index2[0][0]
                        else:
                            start_index_batch[question_index] = -1
                            stop_index_batch[question_index] = -1
                            wrong_count = wrong_count + 1

                        q_length = len(qa['question'].split(' '))
                        question = "".join(qa['question']).replace('?', '')

                        print(ans_index1[0], ans_index2[0], text, question, "raise:", is_wrong)

                        question = question.split(' ')
                        print("question:", question, " Length: ", len(question))

                        if len(question) < 12000:
                            tempArr_q = numpy.zeros((quetion_maxlength), dtype="<U20")

                            for i in range(q_length):
                                tempArr_q[i] = question[i]

                            question_batch[question_index] = tempArr_q
                        else:
                            start_index_batch[question_index] = -1
                            stop_index_batch[question_index] = -1
                            wrong_count = wrong_count + 1


                    question_index = question_index + 1
            paragraph_index = paragraph_index + 1

    print("wrong_count",wrong_count)

    """

    """
    arr_s = numpy.array(["a", "b", "c", "a"])
    arr = numpy.array([[1,2,3], [3,4,5], [4,5,6], [7,8,9]])
    wrong_loc_count = 0
    wrong_count = 0
    loc_diffs = []

    index = 0
    a = 1
    while a == 1:
        index = numpy.where(arr_s == "a")

        if len(index[0]) > 0:
            arr = numpy.delete(arr, index[0][0], 0)
            arr_s = numpy.delete(arr_s, index[0][0], 0)
        else:
            a = 2

    print(arr_s, arr)

    quetion_maxlength = 60
    paragraph_index = 0
    question_index = 0

    numberOfParagraph = 0
    numberOfQuestions = 0

    for article in data['data']:
        for para in article['paragraphs']:
            numberOfParagraph = numberOfParagraph + 1

            for qa in para['qas']:
                for answer in qa['answers']:
                    numberOfQuestions = numberOfQuestions + 1
    batch = numpy.zeros(shape=(numberOfQuestions), dtype=numpy.int)
    paragraph_arr = numpy.zeros(shape=(numberOfParagraph, 825), dtype="<U20")
    question_batch = numpy.zeros(shape=(numberOfQuestions, quetion_maxlength), dtype="<U20")
    start_index_batch = numpy.zeros(shape=(numberOfQuestions), dtype=numpy.int)
    stop_index_batch = numpy.zeros(shape=(numberOfQuestions), dtype=numpy.int)

    for i in range(numberOfQuestions):
        start_index_batch[i] = -1

    for article in data['data']:
        for para in article['paragraphs']:
            para['context'] = para['context'].replace(u'\u000A', '')
            para['context'] = para['context'].replace(u'\u00A0', ' ')
            # para['context'] = para['context'].replace('\"', '')

            context = para['context']
            context = context.replace('.', ' .')
            context = context.replace(',', ' ,')
            context = context.replace('?', ' ?')
            context = context.replace('!', ' !')
            context = context.replace('(', ' ')
            context = context.replace(')', ' ')
            context = context.replace(u'\u2013', ' - ')
            context = context.replace(u'\u2014', ' - ')
            context = context.replace('-', ' - ')
            context = context.replace('\'', ' \' ')
            context = context.replace('\"', '')

            print(context)
            paragraph = numpy.array(context.split(' '))
            print("Max", max, "para:", len(paragraph))

            tempArr_p = numpy.zeros((825), dtype="<U20")
            for i in range(len(paragraph)):
                tempArr_p[i] = paragraph[i]

            paragraph_arr[paragraph_index] = tempArr_p

            is_wrong = 0

            for qa in para['qas']:
                for answer in qa['answers']:
                    answer['text'] = answer['text'].replace(u'\u00A0', ' ')
                    text = answer['text']
                    answer_start = answer['answer_start']
                    if context[answer_start:answer_start + len(text)] == text:
                        if text.lstrip() == text:
                            pass
                        else:
                            answer_start += len(text) - len(text.lstrip())
                            answer['answer_start'] = answer_start
                            text = text.lstrip()
                            answer['text'] = text
                    else:
                        wrong_loc_count += 1
                        text = text.lstrip()
                        answer['text'] = text
                        starts = list(find_all(context, text))
                        if len(starts) == 1:
                            answer_start = starts[0]
                        elif len(starts) > 1:
                            new_answer_start = min(starts, key=lambda s: abs(s - answer_start))
                            loc_diffs.append(abs(new_answer_start - answer_start))
                            answer_start = new_answer_start
                        else:
                            start_index_batch[question_index] = -1
                            stop_index_batch[question_index] = -1
                            print("Raise Exception")
                            print(answer['text'], " : ", qa['question'])
                            # raise Exception()
                            is_wrong = 1
                            wrong_count = wrong_count + 1
                        answer['answer_start'] = answer_start

                    if is_wrong == 0:
                        answer_stop = answer_start + len(text)
                        answer['answer_stop'] = answer_stop

                        # print(answer_start, " : ", answer_stop, context[answer_start:answer_stop])
                        context2 = list(context)
                        context2[answer_start] = '#'
                        context2[answer_stop - 1] = '^'
                        context2 = "".join(context2)
                        context2 = numpy.array(context2.split(' '))

                        text = list(text)
                        text[0] = '#'
                        text[answer_stop - answer_start - 1] = '^'
                        text = "".join(text)
                        text = numpy.array(text.split(' '))

                        ans_index1 = numpy.where(context2 == text[0])
                        ans_index2 = numpy.where(context2 == text[text.size - 1])

                        q_length = len(qa['question'].split(' '))
                        question = "".join(qa['question']).replace('?', '')

                        print(ans_index1[0], ans_index2[0], text, question, "raise:", is_wrong)

                        question = question.split(' ')
                        print("question:", question, " Length: ", len(question))

                        if len(ans_index1[0]) > 0 and len(ans_index2[0]) > 0:
                            if len(question) < 12000:
                                tempArr_q = numpy.zeros((quetion_maxlength), dtype="<U20")

                                for i in range(q_length):
                                    tempArr_q[i] = question[i]

                                question_batch[question_index] = tempArr_q
                                start_index_batch[question_index] = ans_index1[0][0]
                                stop_index_batch[question_index] = ans_index2[0][0]

                                question_index = question_index + 1
                            else:
                                start_index_batch[question_index] = -1
                                stop_index_batch[question_index] = -1
                                wrong_count = wrong_count + 1

                        else:
                            start_index_batch[question_index] = -1
                            stop_index_batch[question_index] = -1
                            wrong_count = wrong_count + 1

            paragraph_index = paragraph_index + 1
    is_loop = 1
    index = 0
    while is_loop == 1:
        if start_index_batch[index] == -1:
            is_loop = 0
        index = index + 1

    print(index, start_index_batch[index - 1], stop_index_batch[index - 1])

    print("wrong_count", wrong_count)
    """

    paragraph = "The role of teacher is often formal and ongoing, carried out at a school or other place of formal education. In many countries, a person who wishes to become a teacher must first obtain specified professional qualifications or credentials from a university or college. These professional qualifications may include the study of pedagogy, the science of teaching. Teachers, like other professionals, may have to continue their education after they qualify, a process known as continuing professional development. Teachers may use a lesson plan to facilitate student learning, providing a course of study which is called the curriculum"
    paragraph = paragraph.replace('.', '')
    paragraph = paragraph.replace(',', '')

    print(paragraph)

    qa = 'Where is school'
    qa = get_Translate("학교에서 선생의 역활은 무엇입니까")

    print("QA!!!!!!!!!!!!!!!:", get_Translate("학교에서 선생의 역활은 무엇입니까"))

    index = 0

    #Index Prediction

    Test = 0
    Case = 0

    qa_model = AoA_Reader.AoA_Reader(is_Para=False)
    print(qa_model.dataset.get_glove_Test('12'))
    print(qa_model.dataset.get_glove_Test('123'))
    print(qa_model.dataset.get_glove_Test('1213'))
    print(qa_model.dataset.get_glove_Test('1223'))
    print(qa_model.dataset.get_glove_Test('1991'))
    print(qa_model.dataset.get_glove_Test('1960'))
    print(qa_model.dataset.get_glove_Test('1055555'))
    print(qa_model.dataset.get_glove_Test('55555'))
    print(qa_model.dataset.get_glove_Test('123555'))
    print(qa_model.dataset.get_glove_Test('99999'))
    print(qa_model.dataset.get_glove_Test('1717'))
    print(qa_model.dataset.get_glove_Test('1522'))
    print(qa_model.dataset.get_glove_Test('1500'))
    print(qa_model.dataset.get_glove_Test('15123123'))
    print(qa_model.dataset.get_glove_Test('151111'))

    print("Setting Complete!")
    #for i in range(10000):
    #    Test = Test + qa_model.dataset.pos_Embedding.check_POS_Tagging(qa_model.dataset.paragraph_arr[i])
    """
    for i in range(qa_model.dataset.numberOf_available_sentence):
        for j in range(qa_model.dataset.p_length):
            if qa_model.dataset.SA_paragraph[i][j] != '' and qa_model.dataset.SA_paragraph[i][j] != '@':
                Test = Test + qa_model.dataset.get_glove_Test(qa_model.dataset.paragraph_arr[i][j])
                Case = Case + 1
                if qa_model.dataset.get_glove_Test(qa_model.dataset.paragraph_arr[i][j]) == 1:
                    print("Wrong Case:", qa_model.dataset.paragraph_arr[i][j])
    """
    print("Result:", Test)

    """
    qa_model = AoA_Reader.AoA_Reader(is_Para=False)

    while index != -1:
        index = int(input("put index"))
        if index != -1:
            a = qa_model.dataset.SA_question[index]
            print(a)

            s = int(qa_model.dataset.SA_start[index])
            t = int(qa_model.dataset.SA_end[index])

            print(qa_model.dataset.SA_paragraph[index])
            print(s, qa_model.dataset.SA_paragraph[index][s])
            print(t, qa_model.dataset.SA_paragraph[index][t])

    qa_model.training_prediction_index(50, is_continue=False, is_Start=True)
    """

    index = 0

    """
        while index != -1:
        index = int(input("put index"))
        if index != -1:
            a = qa_model.dataset.SA_question[index]
            print(a)

            s = int(qa_model.dataset.SA_start[index])
            t = int(qa_model.dataset.SA_end[index])

            print(qa_model.dataset.SA_paragraph[index])
            print(s, qa_model.dataset.SA_paragraph[index][s])
            print(t, qa_model.dataset.SA_paragraph[index][t])

    while index != -1:
        index = int(input("put index"))
        if index != -1:
            a = qa_model.dataset.paragraph_index[index]
            print(qa_model.dataset.paragraph_arr[a])

            s = qa_model.dataset.start_index_batch[index]
            t = qa_model.dataset.stop_index_batch[index]

            print(qa_model.dataset.question_batch[index])

            print(s, qa_model.dataset.paragraph_arr[a][s])
            print(t, qa_model.dataset.paragraph_arr[a][t])

            s = int(qa_model.dataset.SA_start[index])
            t = int(qa_model.dataset.SA_end[index])

            print(qa_model.dataset.SA_paragraph[index])
            print(s, qa_model.dataset.SA_paragraph[index][s])
            print(t, qa_model.dataset.SA_paragraph[index][t])

    #qa_model.training_sto(100)
    #qa_model.training_continue_sto(1000)

    #qa_model.training_continue(100)
    #qa_model.training(50)
    #qa_model.propagate()

    paragraph_ = paragraph.split(' ')

    #a, b = qa_model.get_QA_Answer(para=paragraph, qu=qa)
    #print(paragraph_[a:a+5])
    """


    """
    print(a, b)

    if len(paragraph_) > a:
        print(paragraph_[a])

    if len(paragraph_) > b:
        print(paragraph_[b])
    """

    """

    for i in range(450, 500):
        print(i, "Index")
        print("Question", qa_model.dataset.question_batch[i])
        print("Sta", qa_model.dataset.paragraph_arr[qa_model.dataset.paragraph_index[i]][qa_model.dataset.start_index_batch[i]])
        print("Sto", qa_model.dataset.paragraph_arr[qa_model.dataset.paragraph_index[i]][qa_model.dataset.stop_index_batch[i]])
        print("Sto", qa_model.dataset.paragraph_arr[qa_model.dataset.paragraph_index[i]])
        print()
    """
    #qa_model.propagate()



    #qa_model.training_continue(30)

    #qa_model.training_continue_cpu(50)
    #qa_model.propagate()
    print("Complete!")

