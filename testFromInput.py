# _*_ coding:utf-8 _*_
import sys
import naiveBayes
import numpy


vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam = naiveBayes.getTrainingModel()

while True:
    line = sys.stdin.readline().strip('\n')
    if not line:
        break
    words = naiveBayes.textParser(line)
    result = naiveBayes.intTypeToString(naiveBayes.classify(vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, words))
    print(result)
