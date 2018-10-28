using System;
using HelloMachineLearning.IrisClassification;
using Xunit;

namespace HelloMachineLearningTests.IrisClassification
{
    public class IrisClassifierTests2 : IClassFixture<IrisClassifierTestFixture>
    {
        private readonly IrisClassifierTestFixture _instance;

        public IrisClassifierTests2(IrisClassifierTestFixture instance)
        {
            _instance = instance;
        }

        [Fact]
        public void Test()
        {
            var prediction = _instance.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
            Assert.Equal("Iris-virginica", prediction.PredictedLabels);
        }
    }
}