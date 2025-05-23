# Classification and XAI

This project is an Explainable AI boosted classification model training that can be used for any general labelled dataset.
I have worked on the Brain Tumor and Crack datasets. This doc is in the POV of the crack dataset.

Check this before the code is run: \
Each dataset should have images named by its dataset name, followed by its sequence number in the dataset. 
For example, a dataset named "crack" has its images inside it saved as "crack1.jpg",  "crack2.jpg",..

Folder structure inside project folders\
Each dataset folder should contain images, smoothed_images, segmented_images, and labels.csv\
If you have the dataset images in the required format, these smoothed and segmented images can be saved by following the steps in the code present [here](https://github.com/karthik7712/image_process)

All the dataset folders should be saved inside the "datasets" folder.

Data formatting\
There are some pre-processing techniques to be done to obtain the data that is compatible with the code.


Make sure the folder structure is in the way shown in the image below to ensure the outputs are being saved correctly

Steps to run the code
1) Run train.py in src/training folder - manually enter the dataset name in the main function according to the need.
2) Make sure after the code runs, the classification report and the model path are saved in the outputs folder.
3) Before making the changes for the generate_xai_report.py file run the save_references.py file (change dataset name for different dataset). This saves references file in the dataset folder used to generate the case similarities.
4) Now few little changes are required to be made manually each time you run the XAI code for a new dataset.
a) Use the gradcam.py file outside src folder by running this in the terminal # python generate_gradcam.py --dataset crack
b) change the CLASS_TO_INDEX dictionary in the counterfactual.py file according to the need.
c) Now run the generate_xai_report.py file in src folder.

This is the folder structure to be followed.\
![Screenshot 2025-05-20 105556](https://github.com/user-attachments/assets/eee15475-5ed5-4e55-94b6-3c0f55461942)



![Screenshot 2025-05-23 080117](https://github.com/user-attachments/assets/ea10272d-a60e-4d64-a70d-fa98d65b25a7)




This should run without a problem and automatically save all the output images in the respective folders.




