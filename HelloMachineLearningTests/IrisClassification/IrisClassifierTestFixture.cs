using HelloMachineLearning.IrisClassification;

namespace HelloMachineLearningTests.IrisClassification
{
    public class IrisClassifierTestFixture
    {
        private readonly IrisClassifier _instance = new IrisClassifier();

        public IrisClassifierTestFixture()
        {
            _instance.Train();
        }

        public IrisPrediction Predict(IrisData irisData)
        {
            return _instance.Predict(irisData);
        }
    }
}