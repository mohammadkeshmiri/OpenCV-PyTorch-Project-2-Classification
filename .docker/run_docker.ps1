.\.docker\run_docker_desktop.ps1

# xhost +local:docker
$XSOCK="/tmp/.X11-unix"
$XAUTH="/tmp/.docker.xauth"
$DISPLAY=(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "vEthernet (WSL)").IPAddress +":0"
$LIBGL_ALWAYS_INDIRECT=1

# # Colin : not sure what the line below is doing
# # xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
docker run -it --rm --shm-size 8G --gpus all -e DISPLAY=$DISPLAY -e LIBGL_ALWAYS_INDIRECT=$LIBGL_ALWAYS_INDIRECT -v ${XSOCK}:$XSOCK -v ${XAUTH}:$XAUTH -e XAUTHORITY=${XAUTH} -v ${PWD}:/root/Project2_Kaggle_Competition_Classification python-opencv-pytorch-jupyter:1
# xhost -local:docker