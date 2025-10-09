**Shortened Summary:**  
lietorch.Sim3 encodes 7D Sim(3) poses (3D rotation, 3D translation, 1D uniform scale) via an 8D `.data` tensor for efficient ops: 4D unit quaternion (rotation), 3D translation vector, 1D scale. This adds a quaternion over the minimal 7D for stable SO(3) parameterization.

It's quite telling that the first pose is 

[0, 0, 0, 0, 0, 0, 1, 1]

If we have only translation, rotation, and scale data, we expect: 
0 rotation (represented by a quaternion with vector 0, 0, 0, 1)
0 translation (represeted by a translation vector 0, 0, 0)
1 scale (there's no scale)

The way to organize these is [translation, rotation, scale]

