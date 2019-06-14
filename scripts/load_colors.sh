#/usr/bin/env bash

if [ $# -lt 1 ]; then
  echo Please specify a directory, containing the color config files! >&2;
  exit 1;
fi

colors_dir=$1
param_namespace=$2
loaded_colors=()

for filename in "$colors_dir"/*.yaml; do
  colorname=$(basename "$filename")
  colorname="${colorname%.*}"
  echo Loading color \"$colorname\" config from \"$filename\" to rosparam. >&2
  rosparam load "$filename" "$param_namespace/$colorname"
  if [ $? -eq 0 ]; then
    loaded_colors+=("$colorname")
  fi
done

for colorname in "${loaded_colors[@]}"; do
  echo $colorname
done
