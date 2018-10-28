using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;

namespace HelloMachineLearning.SentimentClassification
{
    public class SentimentClassifier
    {
        private static readonly string DataPath = Path.Combine(Environment.CurrentDirectory, "SentimentClassification", "wikipedia-detox-250-line-data.tsv");

        private static readonly string TestDataPath = Path.Combine(Environment.CurrentDirectory, "SentimentClassification", "wikipedia-detox-250-line-test.tsv");

        //private static readonly string Modelpath = Path.Combine(Environment.CurrentDirectory, "SentimentClassification", "Model.zip");

        private static PredictionModel<SentimentData, SentimentPrediction> _model;

        public void Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(DataPath).CreateFrom<SentimentData>());
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 50, NumTrees = 50, MinDocumentsInLeafs = 20 });
            _model = pipeline.Train<SentimentData, SentimentPrediction>();
            //_model.WriteAsync(Modelpath).Wait();
        }

        public BinaryClassificationMetrics Evaluate()
        {
            var testData = new TextLoader(TestDataPath).CreateFrom<SentimentData>();
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(_model, testData);
            return metrics;
        }

        public IEnumerable<(SentimentData sentiment, SentimentPrediction prediction)> Predict(IEnumerable<SentimentData> sentiments)
        {
            IEnumerable<SentimentPrediction> predictions = _model.Predict(sentiments);
            IEnumerable<(SentimentData sentiment, SentimentPrediction prediction)> sentimentsAndPredictions = sentiments
                .Zip(predictions, (sentiment, prediction) => (sentiment, prediction))
                .ToList();


            return sentimentsAndPredictions;
        }

    }
}
