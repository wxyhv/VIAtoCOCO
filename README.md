# VIAtoCOCO
This repo's function is to convert the json file created by VIA tool to COCO dataset format json file.

Request:
To use this resource, you should install opencv>=3.3.0.

Guideline:

1.Use VIA annotation tools to annotate your image by rectangel shape, then save the annoation file to a specified path.

2.Change the script's VIA annotation file path and image path.

3.Modify the category according to your annotation category.

4.Run the script, and you will get the converted json file in the folder named VIAtoCOCO.

5.If you use other shapes provided by VIA to annotate your images, you need to use the Area class's function in the scirpt.(Tips: the candidate code is commented.)

6.You can split your dataset to train_set and val_set by set the value of 'split' and the related code.
