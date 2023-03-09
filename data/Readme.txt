
Three datasets
--------------

All three datasets are comma-separated text files.


1) wine
This dataset contains tabular data on measured features of white wine.
The dataset contains 11 columns of features, a column of wine "quality" scores in [0,10], and a column to indicate 
wine "type", (such as grape variety). There are three types of wine coded: 1, 2, 3. The quality score could be the 
target of a regression model, and the wine type could be the target of a classification model.
The original data are described here:
	https://www.tensorflow.org/datasets/catalog/wine_quality


2) mnist
This dataset is a subset of the Kaggle MNIST data on images of hand-written digits. 
There are images of four digits: 0, 1, 7, 8. Each digit image is 28*28 pixels scaled into [0,1] by dividing the Kaggle 
images by 255. The dataset contains columns of pixel features (28*28 columns) and a column of digit labels: 0, 1, 7, 8.
The original data are described here:
	http://yann.lecun.com/exdb/mnist/
More on convolutional neural networks:
	https://www.r-bloggers.com/2018/07/convolutional-neural-networks-in-r/
Python users see:
	https://www.tensorflow.org/tutorials/quickstart/beginner
R users see:
	https://tensorflow.rstudio.com/tutorials/beginners/


3) taxi
This dataset is a time series of counts of taxi passengers in New-York city, in half-hour intervals timestamped 
from 2014-07-1 to 2015-01-31. There are some extreme counts that are unusually low or high. These are not labelled 
but have real known causes such as: the New-York marathon, Thanksgiving, Christmas, New Years day, and a snow storm.
The original data are described here:
	https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
More on structural time-series:
	https://blog.tensorflow.org/2019/03/structural-time-series-modeling-in.html


Some general advice
-------------------

The tabular wine data are probably the easiest to model using a neural network, followed by the image data, 
and then the time-series. Whichever dataset you begin with, start with data exploration, summary, and visualization. 
Consider fitting simple regression or classification models. 

Once you have a good idea of what results to expect, fit a baseline neural network that is a standard network model 
for regression or classification, (whichever you think is right), with no probabilistic layers. 
Make sure you have this working and giving you the results you expect before moving on.

Finally, modify your baseline model to make it Bayesian. Evaluate your model and interpret its results. 
The bottom line: does it capture uncertainty? Does the level of uncertainty indicate anomaly?

Write something at every step, just a paragraph or two to help put your final report together.
Don't feel you have to model all three datasets. The marking rubric is:
	Can the report be presented to the client?			60%
	Style and clarity of writing/organisation and presentation	10%
	Background and references					10%
	Executive summary						10%
	Coding and reproducibility					10%



