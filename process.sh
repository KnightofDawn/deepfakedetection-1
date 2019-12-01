# recursively find and split videos
#for i in $(find . -name *.mp4); do y=${i%.mp4}; echo "${y}"; done
for i in $(find . -name *.mp4); do y=${i%.mp4}; mkdir -p "./images/${y%}"; ffmpeg -i "$i" -filter:v "fps=fps=2" -threads 56 "./images/${y%}/images-%08d.jpg"; done
