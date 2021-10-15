# <TODO: NAME>
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
If the code thinks that the point has a lable, the point's label will be a single uppercase letter. If the code thinks the point does not have 
