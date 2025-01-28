# Junk Wax Sports Cards Object Detection Model 🎴⚾

Welcome to the repository **JunkWaxDetection**, hosted by the GitHub organization **JunkWaxData**. This Machine Learning model is designed to identify sports cards from the overproduced "junk wax" era (1985–1996), with exceptional precision and recall metrics. Whether you're a collector, seller, or enthusiast, this model can streamline the identification of cards from various iconic sets.

## Model Overview 🧠

- **Model Version:** Iteration 24
- **Domain:** General (compact) [S1]

### Performance Metrics 📊

- **Precision:** 98.3%
- **Recall:** 95.6%
- **mAP:** 98.6%

### Performance Per Tag 🏷️

| Tag                              | Precision | Recall | Average Precision (AP) |
| -------------------------------- | --------- | ------ | ---------------------- |
| 1982 Donruss                     | 100.0%    | 100.0% | 100.0%                 |
| 1984 Topps                       | 100.0%    | 100.0% | 100.0%                 |
| 1987 Fleer                       | 100.0%    | 95.5%  | 95.5%                  |
| 1987 Topps                       | 100.0%    | 100.0% | 100.0%                 |
| 1988 Donruss                     | 91.3%     | 100.0% | 100.0%                 |
| 1988 Fleer                       | 100.0%    | 76.2%  | 95.2%                  |
| 1988 Fleer Pack                  | 100.0%    | 87.5%  | 100.0%                 |
| 1988 Score                       | 100.0%    | 100.0% | 100.0%                 |
| 1988 Topps                       | 96.0%     | 100.0% | 100.0%                 |
| 1988 Topps Pack                  | 100.0%    | 100.0% | 100.0%                 |
| 1989 Bowman                      | 100.0%    | 94.7%  | 100.0%                 |
| 1989 Donruss                     | 100.0%    | 94.7%  | 100.0%                 |
| 1989 Fleer                       | 100.0%    | 100.0% | 100.0%                 |
| 1989 Score                       | 100.0%    | 100.0% | 100.0%                 |
| 1989 Topps                       | 95.0%     | 90.5%  | 90.5%                  |
| 1989 Topps Pack                  | 100.0%    | 83.3%  | 91.0%                  |
| 1989 Upper Deck                  | 100.0%    | 92.3%  | 100.0%                 |
| 1990 Donruss                     | 100.0%    | 100.0% | 100.0%                 |
| 1990 Fleer                       | 88.6%     | 96.9%  | 95.7%                  |
| 1990 Fleer Pack                  | 100.0%    | 100.0% | 100.0%                 |
| 1990 Leaf                        | 100.0%    | 100.0% | 100.0%                 |
| 1990 Leaf Pack                   | 100.0%    | 80.0%  | 100.0%                 |
| 1990 Topps                       | 100.0%    | 100.0% | 100.0%                 |
| 1990 Upper Deck High Series Pack | 100.0%    | 91.3%  | 100.0%                 |
| 1991 Fleer Ultra                 | 100.0%    | 97.5%  | 100.0%                 |
| 1991 Leaf                        | 100.0%    | 90.9%  | 95.5%                  |
| 1991 Leaf Pack                   | 100.0%    | 75.0%  | 100.0%                 |
| 1991 Topps                       | 92.9%     | 81.3%  | 97.2%                  |
| 1991 Topps Pack                  | 100.0%    | 91.7%  | 100.0%                 |
| 1991 Upper Deck                  | 100.0%    | 89.5%  | 94.7%                  |
| 1991 Upper Deck Low Series Pack  | 100.0%    | 96.0%  | 98.9%                  |
| 1992 Fleer                       | 100.0%    | 100.0% | 100.0%                 |
| 1992 Fleer Ultra                 | 100.0%    | 100.0% | 100.0%                 |
| 1992 Fleer Pack                  | 100.0%    | 50.0%  | 91.7%                  |
| 1992 O-Pee-Chee Premiere         | 100.0%    | 94.7%  | 99.7%                  |
| 1992 Pinnacle                    | 100.0%    | 94.4%  | 99.7%                  |
| 1992 Pinnacle Pack               | 100.0%    | 85.7%  | 100.0%                 |
| 1992 Upper Deck                  | 95.5%     | 87.5%  | 87.5%                  |
| 1992 Upper Deck High Series Pack | 100.0%    | 100.0% | 100.0%                 |
| 1993 Fleer                       | 100.0%    | 95.0%  | 99.8%                  |
| 1993 Fleer Series 1 Pack         | 100.0%    | 100.0% | 100.0%                 |
| 1993 Fleer Series 2 Pack         | 95.0%     | 100.0% | 100.0%                 |
| 1993 Topps                       | 86.4%     | 86.4%  | 98.1%                  |
| 1994 Leaf                        | 95.7%     | 100.0% | 100.0%                 |
| 1994 Pinnacle                    | 100.0%    | 100.0% | 100.0%                 |
| 1994 Score                       | 100.0%    | 100.0% | 100.0%                 |
| 1995 Leaf                        | 100.0%    | 100.0% | 100.0%                 |
| 1995 Select                      | 95.5%     | 100.0% | 100.0%                 |
| 1996 Pinnacle                    | 100.0%    | 100.0% | 100.0%                 |

## Repository Structure 🗂

- `model` - Contains the ONNX and TensorFlow model files.

- `src` - Example projects demonstrating how to use the models.

## How to Use 🛠️

1. Clone this repository to your local machine.
   ```bash
   git clone https://github.com/JunkWaxData/JunkWaxDetection.git
   ```
2. Navigate to the `src` folder for example code in various programming languages.
3. Load the model in your preferred framework and integrate it into your project.

## Example Frameworks 💻

- **Python (ONNXRuntime)**
- **C# (ML.NET)**
- **JavaScript (TensorFlow\.js)**

Feel free to explore the `src` folder for detailed implementation examples. Contributions in other languages are encouraged!

## Contributing 🤝

We encourage community contributions! Whether it's submitting your own example project or improving documentation, we welcome your input.

## License 📄

This project is licensed under the [MIT License](LICENSE). By contributing, you agree to license your work under the same terms.

## Contact 📬

For any inquiries, please reach out to us at [**eric@junkwaxdata.com**](mailto\:eric@junkwaxdata.com).

