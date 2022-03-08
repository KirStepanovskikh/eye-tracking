# Eye-tracking data processing

In this repository you will find: 
- Velocity Threshold Algorithm for Fixation Identification (I-VT) in [eyeTrackingProcessing.py](https://github.com/KirStepanovskikh/eye-tracking/blob/master/eyeTrackingProcessing.py)
- base pipeline how to use I-VT algorithm in [work_example.ipynb](https://github.com/KirStepanovskikh/eye-tracking/blob/master/work_example.ipynb)
- conda environment to easily get all module dependencies in [eye-tracking.yml](https://github.com/KirStepanovskikh/eye-tracking/blob/master/eye-tracking.yml)

**I-VT algorithm**:
1. Calculate the Euclidean distance between consecutive gaze points
2. Convert Euclidean distance to visual angle
3. Calculate point-to-point velocity in degrees per second
4. Label each point below velocity threshold as a fixation point, otherwise saccade point
5. Collapse consecutive fixation points into fixation groups
6. Remove fixation groups with duration below duration threshold
7. Calculate centroid, variation, duration and average velocity of each fixation group 
