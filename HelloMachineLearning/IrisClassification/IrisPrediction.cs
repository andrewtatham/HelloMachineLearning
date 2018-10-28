using Microsoft.ML.Runtime.Api;

namespace HelloMachineLearning.IrisClassification
{
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}