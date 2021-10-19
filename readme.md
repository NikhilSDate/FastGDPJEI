# <TODO: NAME>

This is a fast geometry diagram parsing tool. The main inprovements over minjoonseo's geosolver diagram parser are speed and the ability to detect text in the image and associate text labels with points. 
However, geosolver's diagram parser is often more accurate in detecting lines and circles. 
Pan Lu's InterGPS can detect text more accurately than this system, but it uses the proprietary and paid MathPix API.
This tool does not use any paid software or APIs

This tool is also highly modular which makes it easy to make changes to the various components in order to improve them or customize them for a particular application.
## Quickstart
First clone or download the repository

Import the diagram_graph_builder.py file from the diagram_parser module

Load your diagram image using opencv and call the parse_diagram method on it

The parse_diagram method returns a tuple (interpretation, lines, circles). lines and circles are both dictionaries.
Interpretation is an instance of the Interpretation class. 

Go down to the section on the Interpretation object to learn how to use the information it contains.

## The Interpretation object
Iterating over interpretation in a for loop will yield the points in the interpretation one at a time. 
The point object has attributes containing the label, coordintes of the point and a list of properties. 
Currently, each property is a tuple where the first element is either lieson or centerof and the second element is a reference to either a line of circle. 
If the second element refers to a line, it will be of the form l<x>. The lines dict will contain l<x> as a key. The corresponding value will be the line in the hesse normal form.
If the second element of the property tuple is of the form c<x>, it refers to a circle. You can get the circle in the (x, y, radius) format from the circle dict by passing the circle id as the key.
If the code thinks that the point has a label, the point's label will be a single uppercase letter. If the code thinks the point does not have 

## A basic overview of how this tool works

When you make a call to parse_diagram, first, the text regions in the diagram are detected using connected component analysis.
Next, text is removed from the image and the primitives (lines and circles) are detected.
The circles are detected using the standard hough transform and the lines are detected using the probabilistic hough transform. A smart parameter selection algorithm is used for detecting circles. 
Clustering is now applied to both the lines and circles in order to remove duplicates and improve the detection accuracy.

After the lines and circles are detected, intersection points between the lines and circles and corners are detected. The intersections are corners are then clustered. This provides information about the points in the diagram image.
A convolutional neural network trained on the Chars64k dataset is then used to recognize the character in each text region. Next, the text regions are associated with the points. Finally, the properties for every point are determined.

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
This tool assumes that all points in the diagram will be labelled with uppercase letters. 
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

This project contains the training data from the Seo et al paper annotated with lines, circles, points and point properties.
Another dataset is provided containing significantly more complex diagrams, also annotated in the same format as the aaai dataset

## Results

### Speed

One of the main advantages of this system is that it is significantly faster than geosolver at diagram parsing
On the geosolver training dataset, this tool takes around 55 seconds to parse all diagrams, while the geosolver tool takes approximately 219 seconds
Also, the geosolver tool is often extremely slow on diagrams that are very complex, especially when the diagram conatins a large number of lines or circles.
While this tool is also significantly slower on complex diagrams, it is still much faster than the geosolver diagram parser

### Primitive detection
On the Seo et al. geosolver dataset, this tool achieves a primitive (line and circle) detection precision of 93.7% and recall of 93.33%, which is slightly better than the performance of geosolver on this dataset (without diagram text).
On the more complex dataset this tool achieves a primitive detection precision of 85.5% and a recall of 85.1%

### Property detection

A new evaluation metric is used for this tool: property detection. To compute this metric, first, detected lines and circlees are mathced with ground truth lines and circles.
Then, for named points (points with ground truth labels like A, B, etc.), a query based system is used to match points. First it is determined how many predicted points have the same label.
If there is more than one point with the same label, the ground truth point is not matched with any predicted point. 
Otherwise, it is determined if the single predicted point with the same label within a certain distance of the ground truth point. 
If it does, it is matched with the ground truth point. 

For unnamed points, for every ground truth unnamed point, it is simply determined if there is a predicted unnamed point within a certain distance of the ground truth point.
If there is, that point is matched with the ground truth unnamed point. 

Next, for every property of the ground truth point, it is determined whether the matched predicted point (if it exists) has that property.
Since properties always reference lines or circles, for a property of a predicted point to match a ground truth property, 
both the type of property has to match and the line or circle referenced by the predicted property has to be matched with the line or circle referenced in the ground truth property.






