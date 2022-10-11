# FastGDP

## Introduction
This repository contains the code and data for the paper "An Efficient Approach to Automated Geometry Diagram Parsing" submitted 
to the [Journal of Emerging Investigators](https://emerginginvestigators.org/). The paper
can be found [here](https://emerginginvestigators.org/articles/an-efficient-approach-to-automated-geometry-diagram-parsing).

FastGDP stands for Fast Geometry Diagram Parser. Given a geometry diagram, FastGDP can recognize lines, circles and points. Additionally, FastGDP 
can determine which lines or circle each point lies on and of a point is the center of a circle.
FastGDP can also detect text labels and associate them with points, but this functionality does not currently work very well. 
determined which lines or circle each point lies on and of a point is the center of a circle.
FastGDP is written to be modular which means that it is easy to make changes to the various components in order to improve them or customize them for a particular application.

## Quickstart
- Clone or download the repository

- Import `parse_diagram` and `display_interpretation` from `diagram_parser.diagram_graph_builder` as follows:
    ```python
    from diagram_parser.diagram_graph_builder import parse_diagram, display_interpretation
    ```

- Load your diagram image using OpenCV (the diagram image should be in the BGR format) as follows:
  
  ```python
  img = cv2.imread('<image_path>')
  ```

- Call `parse_diagram`, passing the diagram image as follows:
   ```python
  interpretation, lines, circles = parse_diagram(img)
  ```
  The parse_diagram method returns a tuple `(interpretation, lines, circles)`. 
  `lines` and `circles` are both dictionaries. `interpretation` is an instance of the `Interpretation` class.

- Optionally call `display_interpretation` to visualize the detection result, passing the original diagram image, 
  `interpretation`, `lines`, and `circles`:
    ```python
    display_interpretation(img, interpretation, lines, circles)
    ```

You can find more information on how to use the results of the `parse_diagram` method below

Note that FastGDP does not give good results if the diagram image is either very small or very large. 
The best results are obtained when the width and height of the image are between 150 and 350 pixels

### How to use the results of `parse_diagram`
`parse_diagram` returns a tuple `(interpretation, lines, circles)`.

#### Interpretation
An `Interpretation` object contains the instance variables `points`, `lines`, and `circles`. `points` is a list of instances of the `Point` class.
`lines` and `circles` are both dictionaries. Each key in `lines` is the ID of a line (used to refer to that line in point properties) and the corresponding value is that line in Hesse normal form.
Each key in `circles` is the ID of a circle and the corresponding value of the circle specified in the (x, y, r) format where (x, y) are the coordinates of the center and r is the radius

#### Point
Iterating over interpretation in a `for` loop will yield the points in the interpretation one at a time. 
Each `Point` object has instance variables `label`, which is the label of the point, `coords`, which are the coordinates of the point in the image, and `properties`, which is a list of the properties of the point.
If a label is detected for a point, the label is a single uppercase letter. Otherwise, the label is of the form `p<x>` where `x` is an integer.

#### Point properties

Each point property is a tuple where the first element is either the string "lieson" or the string "centerof" and the second element is the ID of either a line or a circle. 
If the second element refers to a line, the ID is of the form `l<x>`, where `x` is an integer. The `lines` dictionary will contain l<x> as a key and the corresponding value will be the line in Hesse normal form.
If the second element of the property tuple refers to a circle,  the ID is of the form c<x>, where c is an integer. The `circles` dictionary will contain x<x> as a key, and the corresponding value will be the circle in the (x, y, r) format.

## Datasets

Three datasets are provided in this repository. The first is the training data used by geosolver. 
The second is a dataset containing significantly more complex data. The third is the geosolver test 
dataset. The images and annotations for each dataset can be found in the experiments/data directory.

