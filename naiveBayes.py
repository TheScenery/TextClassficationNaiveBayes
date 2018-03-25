# _*_ coding:utf-8 _*_
import re
import numpy

def intTypeToString(type):
    if type == 1:
        return 'spam'
    else:
        return 'ham'

def getTrainingModel():
    f = open('./trainingModel/vocabularyList.txt', 'r', encoding='UTF-8')
    vocabularyList = []
    for line in f.readlines():
        linedata = line.strip('\n')
        vocabularyList.append(linedata)
    pWordsHealthy = numpy.loadtxt('./trainingModel/pWordsHealthy.txt', delimiter='\t')
    pWordsSpamicity = numpy.loadtxt('./trainingModel/pWordsSpamicity.txt', delimiter='\t')
    f = open('./trainingModel/pSpam.txt')
    pSpam = float(f.readline().strip())
    f.close()
    return vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam

def textParser(text):
    regEx = re.compile(r'[^a-zA-Z]|\d')  # only words
    words = regEx.split(text)
    # remove space and to lower case
    words = [word.lower() for word in words if len(word) > 0]
    return words


def lodaEmaildata(fileName):
    f = open(fileName, 'r', encoding='UTF-8')
    labels = []  # 1 for spam, 0 for ham
    emailWords = []
    for line in f.readlines():
        linedatas = line.strip().split('\t')
        if linedatas[0] == 'ham':
            labels.append(0)
        elif linedatas[0] == 'spam':
            labels.append(1)
        # split email words
        words = textParser(linedatas[1])
        emailWords.append(words)
    return labels, emailWords

def createVocabularyList(emailWords):
    vocabularySet = set([])
    for words in emailWords:
        vocabularySet = vocabularySet | set(words)
    vocabularyList = list(vocabularySet)
    return vocabularyList

def setWordsToVecTor(vocabularyList, emailWords):
    vocabMarked = [0] * len(vocabularyList)
    for smsWord in emailWords:
        if smsWord in vocabularyList:
            vocabMarked[vocabularyList.index(smsWord)] += 1
    return vocabMarked

def setWordsListToVecTor(vocabularyList, emailWorkList):
    vocabMarkedList = []
    for i in range(len(emailWorkList)):
        vocabMarked = setWordsToVecTor(vocabularyList, emailWorkList[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList

def trainingNaiveBayes(trainMarkedWords, trainCategory):
    numTrainDoc = len(trainMarkedWords)
    numWords = len(trainMarkedWords[0])

    # calculate the possiblity of a email that is spam P(S)
    pSpam = sum(trainCategory) / float(numTrainDoc)
    
    wordsInSpamNum = numpy.ones(numWords)
    wordsInHealthNum = numpy.ones(numWords)
    spamWordsNum = 2.0
    healthWordsNum = 2.0
    for i in range(0, numTrainDoc):
        if trainCategory[i] == 1:  # is spam
            wordsInSpamNum += trainMarkedWords[i]
            spamWordsNum += sum(trainMarkedWords[i])  
        else:
            wordsInHealthNum += trainMarkedWords[i]
            healthWordsNum += sum(trainMarkedWords[i])
    pWordsSpamicity = numpy.log(wordsInSpamNum / spamWordsNum)
    pWordsHealthy = numpy.log(wordsInHealthNum / healthWordsNum)

    return pWordsSpamicity, pWordsHealthy, pSpam

def classify(vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, testWords):
    testWordsCount = setWordsToVecTor(vocabularyList, testWords)
    testWordsMarkedArray = numpy.array(testWordsCount)
    p1 = sum(testWordsMarkedArray * pWordsSpamicity) + numpy.log(pSpam)
    p0 = sum(testWordsMarkedArray * pWordsHealthy) + numpy.log(1 - pSpam)
    if p1 > p0:
        return 1
    else:
        return 0
