# FastGDP

## Introduction
FastGDP stands for Fast Geometry Diagram Parser. Given a geometry diagram, FastGDP can recognize lines, circles and points. Additionally, FastGDP 
can determine which lines or circle each point lies on and of a point is the center of a circle.
FastGDP can also detect text labels and associate them with points, but this functionality does not currently work very well. 
determined which lines or circle each point lies on and of a point is the center of a circle.
FastGDP is written to be modular which means that it is easy to make changes to the various components in order to improve them or customize them for a particular application.

## Quickstart
First clone or download the repository

Import the `diagram_graph_builder.py` file from the diagram_parser module

Load your diagram image using opencv (diagram image should have 3 channels) and call the parse_diagram method on it

The parse_diagram method returns a tuple (interpretation, lines, circles). lines and circles are both dictionaries.
Interpretation is an instance of the Interpretation class. 

You can find more information on how to use the results of the `parse_diagram` method below

Note that FastGDP does not give good results if the diagram image is either very small or very large. 
The best results are obtained when the width and height of the image are between 150 and 350 pixels

### How to use the results of `parse_diagram`
parse_diagram returns a tuple `(interpretation, lines, circles)`.

#### Interpretation
An `Interpretation` object contains the instance variables `points`, `lines`, and `circles`. `points` is a list of instances of the `Point` class.
`lines` and `circles` are both dictionaries. Each key in `lines` is the ID of a line (used to refer to that line in point properties) and the corresponding value is that line in Hesse normal form.
Each key in `circles` is the ID of a circle and the corresponding value of the circle specified in the (x, y, r) format where (x, y) are the coordinates of the center and r is the radius

#### Point
Iterating over interpretation in a for loop will yield the points in the interpretation one at a time. 
The point object has instance variables `label`, which is the label of the point, `coords`, which are the coordinates of the point in the image, and `properties`, which is a list of the properties of the point.
If a label is detected for a point, the label is a single uppercase letter. Otherwise, the label is of the form p<x> where x is an integer.

#### Point properties

Each point property is a tuple where the first element is either the string "lieson" or the string "centerof" and the second element is the ID of either a line or a circle. 
If the second element refers to a line, the ID is of the form l<x>, where x is an integer. The `lines` dictionary will contain l<x> as a key and the corresponding value will be the line in Hesse normal form.
If the second element of the property tuple refers to a circle,  the ID is of the form c<x>, where c is an integer. The `circles` dictionary will contain x<x> as a key, and the corresponding value will be the circle in the (x, y, r) format.

## An overview of how FastGDP works

When you make a call to parse_diagram, first, the text regions in the diagram are detected using connected component analysis.
Next, text is removed from the image and the primitives (lines and circles) are detected.
The circles are detected using the circle hough transform (CHT) and the lines are detected using the probabilistic hough transform. A parameter selection algorithm is used with the CHT. 
Clustering is now applied to both the lines and circles in order to remove duplicates and improve the detection accuracy.

After the lines and circles are detected, intersection points between the lines and circles and corners are detected. The intersections are corners are then clustered. This provides information about the points in the diagram image.
A convolutional neural network trained on the Chars74k dataset is then used to recognize the character in each text region. Next, the text regions are associated with the points. Finally, the properties for every point are determined.

## The line detector

Before lines are detected, the text is removed from the image using connected component analysis.
Next, the Probabilistic Hough Transform is used to detect lines in the image. Even though the Probabilistic Hough Transform returns the endpoints of the lines, the lines detected do not always cover the entire ground truth line on the diagram.
So, the lines are converted to the Hesse normal form to make them infinite. The main disadvantage of theis system is that intersections between lines can be detected which are actually not in the diagram. 
To combat this problem, only those intersection points are considered which are in a region containing a corner. For more information about this, look in the corner detector section.
the At this stage, there could be some duplicate lines. So, the lines are clustered in the rho-theta space using the DBSCAN clustering algorithm.
Instead of using a standard Euclidean metric, since the rho theta space is essentially in cylindrical coordinates, a different metric is used.
After the lines are clustered, the average of each cluster is calculated. The averaged lines are finally returned.

## The circle detector

To detect circles, minRadius, maxRadius, and param1 are fixed. Next, the highest value of param2 in the range [0, 100] is determined for which at least one circle is detected. This generally tends to give good results unless the image contains multiple circles with very large differences in radii.
Even after selecting the best value of param2, duplicate circles might be detected. So, the centers of the circles are clustered, and for each cluster, the centers and radii are both averaged. 
The averaged circles are finally returned.

## The corner detector

The standard Harris Corner Detector is used to detect corners. The Harris response map is dilated and the connected components in the response map are determined.
The connected components in the corner response map are now determined and the centroid of each component is detected as a corner.
The original (non-dilated) response from the corner detector is now copied and dilated again with a larger kernel size than in the first dilation.
This dilated response map is used to determine if a detected intersection point is indeed an intersection in the image. The intersection is accepted if it lies in a corner region in the harris corner map.

## The text detector

To detect text regions in the image, the image is thresholded, and the connected components in the thresholded image are determined, Next, the connected components are filtered by size to filter out the main diagram component and salt and pepper noise.
The text regions are then padded to make them square and are then fed into a convolutional neural network trained on the Chars64k dataset. The netword is trained to recognize lowercase letters, uppercase letters, and digits. 
FastGDP assumes that all points in the diagram will be labelled with uppercase letters. 
Due to the nature of the Chars64k dataset, the network often confuses the uppercase and lowercase forms of some letters. To combat this issue, first, the confusion matrix of the model on the validation data is computed.
Next, when recognizing the letter in each text region, if the letter detected by the network is a lowercase letter that is frequently confused by the model with its uppercase form, the uppercase form of the letter is returned.
Here, 'frequently confused' means that the entry for the lowercase letter-uppercase letter pair in the confusion matrix is above a certain threshold.
The network has been trained to assume that each text region is around 12x12 in size. If the input image has high resolution, detection accuracy can be increased by using a network trained on larger images in place of the current network,
## Putting it all together

To determine the locations of the points in the image, the intersection point or points for every line-line, line-circle, or circle-circle pair is determined.
The intersections are filtered based on the harris corner response map as described in the corner detector section.
The intersections are now put together with the corner points detected (using the method described in the corner detector section) and are clustered using the DBSCAN algorithm.
Next, the text regions are associated with the clustered points using a greedy method. The text regions are first sorted based on the text detector's confidence that the text region is an uppercase letter.
Then, each text region is associated with the point that maximizes the following score.

Do nothing: score = 0

If point has more than one label: score = -infinity

if point contains at least one intersection: score = offset - distance between centroid(intersection points) and text region

else: score = offset - distance between centroid(corners) and text region

offset is a parameter that controls the distance beyond which doing nothing is better than associating the text region with the point.

Finally, the properties for each point are determined

## Datasets

Three datasets are provided in this repository. The first is the training data used by geosolver. 
The second is a dataset containing significantly more complex data. The third is the geosolver test 
dataset. 
