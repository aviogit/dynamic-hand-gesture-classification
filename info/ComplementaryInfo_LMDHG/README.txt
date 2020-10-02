First, you may note that we have published a paper describing this dataset.
In this paper we also provided some preliminary results that you can compare to. Here is the reference:

Boulahia, S. Y., Anquetil, E., Multon, F., & Kulpa, R.
Dynamic hand gesture recognition based on 3D pattern assembled trajectories.
In 7th IEEE International Conference on Image Processing Theory, Tools and Applications (IPTA 2017), (pp. 1-6). 


Second, two files are attached to this dataset. 
The first file "ArticulationsOrder.txt" contains the order of the 46 articulations recorded in our dataset.
In fact, at each frame we record the 3D locations (x,y,z) of 23 articulations of each hand giving a total of 138 dimension at each frame.
As specified in this file, we recorded data of Left hand and then that of Right hand. I provided an example for each coordinate in this file.


The second file is called "demoProject.m" which is a matlab file that allows you to parse and visualize the gesture.
You can then modify it for your own needs.

Please note that as you download the dataset, there are always two kind of files a "DataFileX.mat" and  "DataFileX.txt".
These two files contain the same data but are saved in different format.
I suggest that you use the .mat along with matlab to extract the data as I provide you with some preliminary code to parse each frame.
