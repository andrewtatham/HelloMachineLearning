using HelloMachineLearning;
using Xunit;

namespace HelloMachineLearningTests
{
    public class IrisClassifierTestFixtureXUnit
    {
        [Fact]
        public void Test()
        {
            IrisClassifier.Main(null);
        }
    }
}