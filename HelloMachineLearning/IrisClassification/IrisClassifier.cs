using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;

namespace HelloMachineLearning.IrisClassification
{
    public class IrisClassifier
    {
        private readonly IHostEnvironment _env = new LocalEnvironment();
        private TransformerChain<KeyToValueTransform> _model;

        public void Train()
        {

            // If working in Visual Studio, make sure the 'Copy to Output Directory'
            // property of iris-data.txt is set to 'Copy always'
            string dataPath = "IrisClassification/iris.data.txt";
            var reader = new TextLoader(_env,
                new TextLoader.Arguments()
                {
                    Separator = ",",
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoader.Column("SepalLength", DataKind.R4, 0),
                        new TextLoader.Column("SepalWidth", DataKind.R4, 1),
                        new TextLoader.Column("PetalLength", DataKind.R4, 2),
                        new TextLoader.Column("PetalWidth", DataKind.R4, 3),
                        new TextLoader.Column("Label", DataKind.Text, 4)
                    }
                });

            IDataView trainingDataView = reader.Read(new MultiFileSource(dataPath));

            // STEP 3: Transform your data and add a learner
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training.
            // Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
            // Convert the Label back into original text (after converting to number in step 3)
            var pipeline = new TermEstimator(_env, "Label", "Label")
                .Append(new ConcatEstimator(_env, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(new SdcaMultiClassTrainer(_env, new SdcaMultiClassTrainer.Arguments()))
                .Append(new KeyToValueEstimator(_env, "PredictedLabel"));

            // STEP 4: Train your model based on the data set  
            _model = pipeline.Fit(trainingDataView);
        }

        public IrisPrediction Predict(IrisData data)
        {
            // STEP 5: Use your model to make a prediction
            // You can change these numbers to test different predictions
            return _model.MakePredictionFunction<IrisData, IrisPrediction>(_env).Predict(data);
        }
    }
}