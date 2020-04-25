import Cocoa
import CreateML

//1. MLDataTable로 데이터를 긁어옴
//2. training data, testing data를 만들어서 random 하게 데이터 setting
//3. sentimentClassifier로 training data initialization
//4. evaluationMerics를 만들어서 testing data를 넣는다.
//5. evaluation Accuracy를 계산
//6. meta data를 추가해준다.
//7. MLModel을 만든다.

/*
 더 간략하게
 1. MLDataTable로 데이터를 만든다. -> Training Data, Testing Data
 2. Create Classifier with Training data
 3. Create Testing Data with Testing Data
 4. Classifier.prediction (Classifier.write)
 */


//1. MLDataTable로 데이터를 긁어옴
let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/grosso/Desktop/Programming_source/Learning/ios/NLP_swift/twitter-sanders-apple3.csv"))

//2. training data, testing data를 만들어서 random 하게 데이터 setting
//Create more than one constant
//random split
//기본적인 데이터 세팅 - random하게 split한다.
let(trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)

//3. sentimentClassifier로 training data initialization
//TextClassifier
let sentimentClassifier: MLTextClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "class")

//4. evaluationMerics를 만들어서 testing data를 넣는다.
//Evaluation Metrics
let evaluationMetrics: MLClassifierMetrics = sentimentClassifier.evaluation(on: testingData, textColumn: "text", labelColumn: "class")

//5. evaluation Accuracy를 계산
let evaluationAccuracy: Double = (1.0 - evaluationMetrics.classificationError) * 100

//6. meta data를 추가해준다.
//My Meta data
let metadata = MLModelMetadata(author: "Paige Shin", shortDescription: "A model trained to classify sentiment on Tweets", version: "1.0")

//7. MLModel을 만든다.
try sentimentClassifier.write(to: URL(fileURLWithPath: "/Users/grosso/Desktop/Programming_source/Learning/ios/NLP_swift/TweetSentimentClassifier.mlmodel"))

//result - Neg
try sentimentClassifier.prediction(from: "@Apple is a terrible company!")

//result - Pos
try sentimentClassifier.prediction(from: "I just found the best restarant ever, and it's @DuckandWaffle!")

//result - Neutral
try sentimentClassifier.prediction(from: "I think @ColcaCola ads are justt ok.")
