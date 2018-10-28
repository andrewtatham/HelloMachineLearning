using Microsoft.ML.Runtime.Api;

namespace HelloMachineLearning
{
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}