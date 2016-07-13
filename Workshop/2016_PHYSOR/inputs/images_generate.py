#!/bin/env python3

print("set xrange [0.0:5.0]")
print("set yrange [0.0:5.0]")
print("set term png")
number_of_timesteps = 100
for j in range(number_of_timesteps):
  time = j*0.1
  timeStr = str(round(time,1))
  filename = "timeSet_"+ timeStr
  print('set output "image'+("%03d" % j)+'.png"')
  print("plot",'"'+filename+'"',' using 2:3 title "Time '+timeStr+'"')
  #print("pause -1")


# ./images_generate.py  > plot_script
# gnuplot plot_script
# oggSlideshow image*.png -o slideshow.ogg -t p -l 0.1 -s 640x480
# ffmpeg -i slideshow.ogg slideshow.mpg
