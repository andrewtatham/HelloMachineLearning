using System;
using HelloMachineLearning;
using NUnit.Framework;


namespace HelloMachineLearningTests
{
    [TestFixture]
    public class IrisClassifierTestFixtureNUnit
    {
        [Test]
        public void Test()
        {
            IrisClassifier.Main(null);
        }
    }
}
