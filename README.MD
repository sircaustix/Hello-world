The Compile Steps remain the same as previously.
To run as a batch executable 
./main -r Reference Image Path 
       -d directory containing the images to be normalized 
       -o output directory to write the normalized images
       -l all to convert all files or comma separated list of files like img1.jpg,img2.jpg
       -m normalization mode, m(Macenko), v(Vahadane) or r(Reinhard)
To get help:
./main -h       
 NOTE: No space should follow the comma in case of a list of files. 
 
 Ex: /main -r 10_134.jpeg -d data -o tmp3 -l 52_86.jpeg,49_68.jpeg  -m m
