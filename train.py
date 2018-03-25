# _*_ coding:utf-8 _*_
import naiveBayes
import numpy

trainingFileName = './emails/trainingData/SMSCollection.txt'

labels, emailWords = naiveBayes.lodaEmaildata(trainingFileName)

vocabularyList = naiveBayes.createVocabularyList(emailWords)

vocabMarkedList = naiveBayes.setWordsListToVecTor(vocabularyList, emailWords)

trainMarkedWords = numpy.array(vocabMarkedList)

pWordsSpamicity, pWordsHealthy, pSpam= naiveBayes.trainingNaiveBayes(trainMarkedWords, labels)

fpSpam = open('./trainingModel/pSpam.txt', 'w')
spam = pSpam.__str__()
fpSpam.write(spam)
fpSpam.close()

fw = open('./trainingModel/vocabularyList.txt', 'w')
for i in range(len(vocabularyList)):
    fw.write(vocabularyList[i] + '\n')
fw.flush()
fw.close()

numpy.savetxt('./trainingModel/pWordsSpamicity.txt', pWordsSpamicity, delimiter='\t')
numpy.savetxt('./trainingModel/pWordsHealthy.txt', pWordsHealthy, delimiter='\t')