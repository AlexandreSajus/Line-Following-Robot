# Line Following Robot
Winning an autonomous line following robot competition at CentraleSupélec. The robot is capable of navigating lines, intersections and roundabouts

![](media/robot.png)

## Demo
**Turning at an intersection**

![](media/intersection.gif)

**Detecting an obstacle and changing the route to avoid it**

![](media/obstacle.gif)

**Navigating a roundabout**

![](media/roundabout.gif)

## Utilities
I have included many useful functions to reproduce the result in utilities.py such as:

**image_preprocessing**: image preprocessing to isolate white lines in an image

![](media/image_preprocessing.png)

**direction**: finds the main line in an image and returns its position in pixels

![](media/direction.png)

**turn_detection**: detects left and right turns if they are close

![](media/turn_detection.png)

**detect_red**: detects red in an image

![](media/detect_red.png)
