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

