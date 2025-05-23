# Classification and XAI

This project is Explainable AI helped classification model which can be used for general labelled datasets.
I have worked on Brain Tumor and Crack datasets. Two completly unrelated datasets.

Check this before code is run: \
Each dataset should have images named by its dataset name followed by its sequence number in the dataset. 
For example a dataset named "crack" has its images inside it saved as "crack1.jpg", "crack2.jpg",..

Folder Structutre inside project folders\
Each dataset folder should contain images, smoothed_images, segmented_images and labels.csv\
If you have the images in the required format, this step can be achieved by following steps in the code present [here](https://github.com/karthik7712/image_process)

All the dataset folders should be saved inside the "datasets" folder.

Data formatting\
There are some pre processing techniques to be done to obtain the data that is compatible to the code.


Make sure the folder Structutre is in the way shown in the image below to ensure the outputs are being saved correctly

steps to run the code
1) Run train.py in src/training folder - manually enter the dataset name in the main function according to the need.
2) Make sure after the code runs, the classification report and the model path are saved in the outputs folder.
3) Before making the changes for the generate_xai_report.py file run the save_references.py file (change dataset name for different dataset). This saves references file in the dataset folder used to generate the case similarities.
4) Now few little changes are required to be made manually each time you run the XAI code for a new dataset.
a) Use the gradcam.py file outside src folder by running this in the terminal # python generate_gradcam.py --dataset crack
b) change the CLASS_TO_INDEX dictionary in the counterfactual.py file according to the need.
c) Now run the generate_xai_report.py file in src folder.

This should run without a problem and save all the output images in the respective folders automatically.




