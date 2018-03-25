# _*_ coding:utf-8 _*_
import numpy
import naiveBayes

vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam = naiveBayes.getTrainingModel()

testFilename = './emails/testingData/testingEmails.txt'
labels, testEmailWordsList = naiveBayes.lodaEmaildata(testFilename)

testResult = []
for i in range(len(testEmailWordsList)):
    testEmailWords = testEmailWordsList[i]
    realType = naiveBayes.intTypeToString(labels[i])
    classifyType = naiveBayes.classify(vocabularyList, pWordsSpamicity, pWordsHealthy, pSpam, testEmailWords)
    predictType = naiveBayes.intTypeToString(classifyType)
    print("real：", realType, "predict：", predictType)
    testResult.append("real：" + realType + "  " + "predict：" + predictType)

fw = open('./emails/testingData/result.txt', 'w', encoding='UTF-8')
for i in range(len(testResult)):
    fw.write(testResult[i] + '\n')
fw.flush()
fw.close()
