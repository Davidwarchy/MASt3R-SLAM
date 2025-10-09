There's this problem where we want to have a robot navigate automously in an environment. 

The model we currently have is able to get the point cloud map of the environment. 

There's is also a floor (so the floor is part of the point cloud). 

A natural guess of going about this is to project the point cloud to a 2d plane, and then use this 2d plane to pick frontiers. The robot can then turn and move towards the frontiers. This way, we can explore the map. 

The problem is that we don't know which plane the right plane is in the point cloud. We don't know what plane the robot is trravelling on. We can use the pose information (and for this we'd need to do some random exploration in order to discren which plane is being used for navigation) to get a plane. Ie, we can derive 3d coordinates and then see what is the plane being used. Then we can project what's above on this plane. Again, we don't know what above or below is. All this modifications introduce a great deal of handcrafting, which affects the autonomy of the robot. If a human being is needed to program these things, this undermines the robots autonomy and capabililty to navigate from near tabula rasa. Additionally, if we tried to project the floor, it might show occupied at place which are navigable in reality. So this is an interesting problem. 

So what do we do? 

I think that the solution could be try and implement the handcrafted system. Then we can then implement the system without handcrafting. This seems like an interesting research direction. 

We could get the plane along which the robot travels by finding the pca components. But how would we know up and down? 

Also flatness of our environment is unrealistic. It might be interesting to extend this to places with undulating floors and grounds. But this is an issue for another day. Right now, we want a system that can run in and of its own on a building floor, which is typically flat. But we want to remember this important detail for the sake of future designs. Or we could consider only objects at the camera level?? Maybe this could work. Let's try that: have some random motion (maybe 50 steps or so), to establish the plane of movement. Once that is done, we want to then select a random frontier, turn towards that frontier, and then move towards the frontier. All the while, we want to allow master-slam to do mapping. If we are stuck, we want to do a LFLFLF sequence, then select a new frontier. 

We want to have another object that does all these things, import it to master slam so that we don't interfere with the main flow of the system. It will contain code for: 
- occupancy grid 
- path planning 
- pid control of turning and such 
- frontier selection 

## MASt3r-SLAM 
Here's how MASt3r-SLAM works. You give it a camera feed only, it returns a 3d reconstruction in the form of a point cloud (PC). Works pretty well, but needs a human being to guide it. It would be desirable if a robot was able to explore and map on its own without human supervision. Most remarkably, it uses only rgb images as input. 

## Making MASt3r-SLAM autonomous (for active slam)
* First move randomly, to get a **plane to project to**. We can extend this later, updating the projection plane with subsequent poses. Take note that there is handcrafting here. 
* What about direction of projection? is it above the plane or below the plane? We could simply check around the area the robot is at. It's likely that if we are in an unroofed place, then the ground below will be mapped (ie, will have point clouds). It's good to take note that this is handcrafting as well. 
* * If the region is roofed, the point clouds will be further away than those on the ground, so we can use this human-knowledge to map and explore. We could make a grid from a plane at the camera level (at the camera plane - the plane parallel to the ground that cuts through the camera). Note that if our environment is not flat this will not work. 
* * Instead of relying on these sort of things, we can try approach this from a more simpler perspective. Ie, imbue the robot with forward movements to certain objects. It will likely be moving forward... This is an incomplete thought. 

* Where are we in the point cloud at any given moment. Given a pose, can we show the path we took in the point cloud? I think we need to know where we are in the point cloud in order to make the desired cut. 

### Marking free regions 
* How do we know hat a region is free? It's usually obvious when using Lidar, but less obvious with RGB images. We can simply check whether we have point clouds where we are facing right now. MASt3r-SLAM doesn't keep track of free areas, and there's no distinction between unknown and free areas from just the point cloud. 

Perhaps one thing that can lead us to better (although it's super human crafted) is sort of keeping note of the direction of free zones. Then we can note from where they were recorded (ie, from approximately). Then we can preserve these free zones as the point cloud updates. 

### 

## Simpler experiment
* Maybe start with a simpler environment 
* We need to be patient enough learning point clouds and Sim(3) poses 
* How do we update the occupancy grid if the point cloud keeps being refined (by optimization process, which are neccessary for accurate maps)

## More Elegance - Less Handcrafting
- The point of not handcrafting
- Are there robots that are human programmable (this seems to be the most commercially sensible option for LifeOS. I think that a balance need be stricken between commercial sensibility and the dream, and the dream shouldn't be given over to commercial sensibility). 
- How would we go about creating a system that removes the handcrafting and would work in
- - Undulating terrain (ground that isn't flat)
- - 

## Later Developments
I keep having this reccuring feeling that approach/avoidance is the key to the whole thing, but I don't know the specific details. I think that I want to complete the autonomous mapping first before doing stuff with

