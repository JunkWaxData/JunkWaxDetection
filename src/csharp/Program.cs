using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace JunkWaxDetection.csharp
{
    /// <summary>
    ///     Example Program showing how to use the JunkWaxDetection ONNX Model in C#
    ///     utilizing the Microsoft.ML.OnnxRuntime NuGet Package
    /// </summary>
    internal class Program
    {
        // Required Image Size for the Model
        const int RequiredHeight = 320;
        const int RequiredWidth = 320;

        // Model and Label Paths
        private const string modelName = "model.onnx";
        private const string modelLabels = "labels.txt";

        // Labels
        private static List<string> _labels = [];

        static void Main(string[] args)
        {
            //Load input image from command line
            var image = SKBitmap.Decode(args[0]);

            //Load the labels
            _labels = File.ReadAllLines(modelLabels).ToList();

            //Preprocess the image
            var resizedImage = PreprocessImage(image);

            //Get the tensor for the image
            var tensor = GetTensorForImage(resizedImage);

            //Get Predictions
            var predictions = GetPredictions(tensor);

            //Write Predictions to Console
            Console.WriteLine("Detected Labels:");
            foreach (var prediction in predictions)
            {
                Console.WriteLine($"{prediction.Item1} - {Math.Round(prediction.Item2 * 100, 2)}%");
            }
        }

        /// <summary>
        ///     Resizes and process images for use by the Onnx Model
        ///
        ///     For Azure Custom Vision, our model is Tensorflow Lite 2.1 and the input images must be:
        ///     - 320x320
        ///     - 3 channels (RGB)
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        private static SKBitmap PreprocessImage(SKBitmap image)
        {
            //Resize the image
            var resizedImage = new SKBitmap(RequiredWidth, RequiredHeight);
            using var surface = SKSurface.Create(new SKImageInfo(RequiredWidth, RequiredHeight));
            using (var paint = new SKPaint())
            {
                surface.Canvas.DrawBitmap(image, SKRect.Create(RequiredWidth, RequiredHeight), paint);
            }
            surface.Snapshot().ReadPixels(resizedImage.Info, resizedImage.GetPixels(), resizedImage.RowBytes, 0, 0);

            return resizedImage;
        }

        /// <summary>
        ///     Gets the Tensor for the image
        ///
        ///     Goes Pixel by Pixel and sets the RGB values in the tensor
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        private static Tensor<float> GetTensorForImage(SKBitmap image)
        {
            //Convert the image to a tensor
            var tensor = new DenseTensor<float>([1, 3, RequiredHeight, RequiredWidth]);
            for (var y = 0; y < RequiredHeight; y++)
            {
                for (var x = 0; x < RequiredWidth; x++)
                {
                    var pixel = image.GetPixel(x, y);
                    tensor[0, 0, y, x] = pixel.Red;
                    tensor[0, 1, y, x] = pixel.Green;
                    tensor[0, 2, y, x] = pixel.Blue;
                }
            }
            return tensor;
        }

        /// <summary>
        ///     Generates Predictions from Model
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private static List<Tuple<string, float, SKRect>> GetPredictions(Tensor<float> input)
        {
            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("image_tensor", input) };

            // Run inference
            var results = new InferenceSession(modelName).Run(inputs);

            var resultsDict = results.ToDictionary(x => x.Name, x => x);
            var detectedBoxes = resultsDict["detected_boxes"].AsTensor<float>();
            var detectedClasses = resultsDict["detected_classes"].AsTensor<long>();
            var detectedScores = resultsDict["detected_scores"].AsTensor<float>();

            var numBoxes = detectedClasses.Length;

            //Build Results
            var output = new List<Tuple<string, float, SKRect>>();


            for (var i = 0; i < numBoxes; i++)
            {
                var score = detectedScores[0, i];
                var classId = detectedClasses[0, i];
                var x = detectedBoxes[0, i, 0];
                var y = detectedBoxes[0, i, 1];
                var x2 = detectedBoxes[0, i, 2];
                var y2 = detectedBoxes[0, i, 3];

                //Convert to pixel values for the original input image
                x *= RequiredWidth;
                y *= RequiredHeight;
                x2 *= RequiredWidth;
                y2 *= RequiredHeight;

                output.Add(new Tuple<string, float, SKRect>(_labels[(int)classId], score, new SKRect(x, y, x2, y2)));
            }

            return output;
        }
    }


}
