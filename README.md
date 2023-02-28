# AML-volcano

run **base_line_combined_server.py** in the server to get the result of data after the first feature selection method(FRESH).

run **pipeline-final.ipynb** to get the result
It is divided into several parts:
1. Library dependency, the library needed in the code.
2. Create label and combine the separate data: create labels with "new_data.csv", the manul recording with given year, period in the year, step(gap), length, and the sampling rate.
If you have already created one, there is no need to generate a new file (If you have the file, you can ignore this part).
3. Additional Attribute, the function for Four Types of Samples, run it before the pipeline.
4. RACOG, the function for RACOG, if you want to use it, you should implement it and replace SMOTE with it.
5. Boruta, the function for Boruta, run it before the pipeline.
6. Automated Tuned RF, the function for Automated Tuned Random Forest, run it before the pipeline.
7. Run pipeline, the part for loading the data and label, and for running the pipeline. The input data is from the file base_line_combined_server.py, which generates the
data after the first feature selection method(FRESH). The output is the F1 score and Accuracy. 
