using Microsoft.ML.Runtime.Api;

namespace HelloMachineLearning.SentimentClassification
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Sentiment;
    }
}