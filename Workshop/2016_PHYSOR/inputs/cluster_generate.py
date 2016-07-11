#!/bin/env python3

import random
random.seed(64616)

number = 100

cluster_starts = 3

cluster_x = [2.0, 4.0, 4.5]
cluster_y = [2.0, 1.0, 4.0]

cluster_dx = [0.1, 0.05, -0.05]
cluster_dy = [0.1, 0.2, -0.2]

point_x = [None]*number
point_y = [None]*number

point_dx = [None]*number
point_dy = [None]*number

point_file = [None]*number

for i in range(number):
  point_file[i] = open("dataSet_"+str(i)+".csv","w")
  point_file[i].write("time,x,y\n")
  cluster = random.randint(0,cluster_starts-1)
  xy_sigma = 0.1
  point_x[i] = cluster_x[cluster] + random.gauss(0.0, xy_sigma)
  point_y[i] = cluster_y[cluster] + random.gauss(0.0, xy_sigma)
  dxy_sigma = 0.01
  point_dx[i] = cluster_dx[cluster] + random.gauss(0.0, dxy_sigma)
  point_dy[i] = cluster_dy[cluster] + random.gauss(0.0, dxy_sigma)

  #print(i,point_x[i],point_y[i],point_dx[i],point_dy[i])

number_of_timesteps = 100
for j in range(number_of_timesteps):
  time = j*0.1
  time_file = open("timeSet_"+str(round(time,1)),"w")
  for i in range(number):
    cx,cy = point_x[i]+time*point_dx[i],point_y[i]+time*point_dy[i]
    print(time,cx,cy,sep=",",file=point_file[i])
    print(i,cx,cy,file=time_file)
    print(time,i,cx,cy)
  time_file.close()
