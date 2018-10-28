using System;
using System.Collections.Generic;
using HelloMachineLearning.SentimentClassification;
using Xunit;

namespace HelloMachineLearningTests.SentimentClassification
{
    public class SentimentClassificationTests
    {
        private readonly SentimentClassifier _instance = new SentimentClassifier();

        [Fact]
        public void Test()
        {
            _instance.Train();
            
            var metrics =  _instance.Evaluate();
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");


            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Please refrain from adding nonsense to Wikipedia."
                },
                new SentimentData
                {
                    SentimentText = "He is the best, and the article should say that."
                }
            };
            var sentimentsAndPredictions = _instance.Predict(sentiments);
            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Negative" : "Positive")}");
            }
            Console.WriteLine();


        }
    }
}
