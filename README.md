# Fire Classification Project

This project focuses on developing a machine learning model to detect and classify fire occurrences. By leveraging deep learning techniques, the model aims to distinguish between fire types using both Image and sensor data, providing a tool for early fire detection and mitigation. A final year project for the Federal University of technology, Minna, Niger state.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. It's recommended to use a virtual environment to manage dependencies.

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/CerealJosh/Fire-Classification-Project.git
   cd Fire-Classification-Project
   ```

2. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Create the Models Folder**:
   ```bash
   mkdir Trained_Models
   ```

## Usage

### Training the Model

To train the image model, prepare your dataset in the `Image_Data` directory, ensuring you have separate folders for different fire type images. Then, execute the training scripts located in the `Image_Models` directory.

Similarly, To train the data models, execute the training scripts in the `Data_Models` directory.

### Running Inference

For running inference on new images, use the `Image_Loader.py` or `Image_Loader.ipynb` script:

```bash
python Image_Loader.py
```

This will output the colour of the flame in the fire image.

Run the `inference.py` or `inference.ipynb` to perform classification on the values provided in `data.json`.

```bash
python inference.py
```
This will output the infered class of the fire.

## Dataset

The dataset comprises images categorized into three different fire colours. For more comprehensive datasets, consider exploring open-source repositories such as the [Fire Detection/Classification dataset by FYPFireRooster](https://universe.roboflow.com/fypfirerooster/fire-detection-classification) for other fire image type classifications suitable for training and evaluation.

The second dataset is comprised of collected numerical sensor values through real-world data collection. It is recommended to collect your own data, although our own data is also provided in the `Data_Models` directory.

## Model Architecture

The Image project utilizes convolutional neural networks (CNNs) for image classification. Specifically, pre-trained architectures like VGG-16 have been employed due to their effectiveness in image recognition tasks. For instance, similar projects have successfully implemented VGG-16 for forest fire classification .

The Data project employs a wide variety of approaches ranging from logistic regression to CNNs.

## Results

Evaluation metrics such as accuracy, precision, recall, and F1 score are used to assess model performance. Detailed results and performance graphs are available in the `Results` directory.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your enhancements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Authors
Eustace M. Dogo,
Joshua Onu and
Terence Bentem
