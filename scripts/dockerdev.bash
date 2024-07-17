# MIT License
#
# Copyright (c) 2019 Brian DuSell
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---
#
# dockerdev.bash
#
# A library of bash functions to facilitate using Docker containers as local
# development environments.
#
# Repository: https://github.com/bdusell/dockerdev
# Report bugs at: https://github.com/bdusell/dockerdev/issues
#
# This file defines functions for managing Docker containers that are meant
# to serve as local development environments. Typically the source code of
# your application will be bind-mounted into the container so that you have
# write access to the source code from within the container (which is useful
# for things like package lock files).
#
# In addition to providing functions for managing single containers, this
# file also includes functions for managing *stacks* of Docker containers
# (created with `docker deploy`) that are meant to serve as local development
# environments. This is necessary if, for example, you want to use a container
# with Docker secrets.
#
# You should source this file in another bash script like so:
#
#     . dockerdev.bash
#
# or
#
#     source dockerdev.bash
#
# You will mainly be interested in these functions:
# * dockerdev_ensure_dev_container_started
#     Ensures that a dev container is started with a certain image.
# * dockerdev_run_in_dev_container
#     Runs a command in a dev container.
# * dockerdev_run_in_dev_stack_container
#     Runs a command in a dev container that is part of a stack.

# Start of include guard.
if [[ ! ${_DOCKERDEV_INCLUDED-} ]]; then
_DOCKERDEV_INCLUDED=1

DOCKERDEV_VERSION='0.5.3'

# dockerdev_container_info <container-name>
#   Get the image name and status of a container.
dockerdev_container_info() {
  local container_name=$1
  # The ^/ ... $ syntax is necessary to get an exact match. This does not
  # appear to be documented anywhere.
  # See https://forums.docker.com/t/how-to-filter-docker-ps-by-exact-name/2880
  docker ps -a -f name="^/$container_name\$" --format '{{.Image}} {{.Status}}'
}

# dockerdev_start_new_container <container-name> <image-name> \
#     [<docker-run-flags>...]
#   Start a new container with a certain image.
dockerdev_start_new_container() {
  local container_name=$1
  local image_name=$2
  shift 2
  echo 'Starting new container' 1>&2 &&
  docker run -d "$@" --name "$container_name" "$image_name"
}

_dockerdev_add_user() {
  # NOTE: The addgroup and adduser commands are not portable across Linux
  # distributions. However, it is best to use them because they automatically
  # set up a home directory, and some tools balk if there is no home directory.
  local userid
  local groupid
  local groupname
  userid=$(id -u "$USER") &&
  groupid=$(id -g "$USER") &&
  # On Mac, the group name might not be valid in Linux, so we normalize it.
  groupname=$(id -gn "$USER" | tr '[A-Z]' '[a-z]' | sed 's/[^-a-z0-9_.@]/-/g') &&
  echo "
    if addgroup --help 2>&1 | head -1 | grep -i busybox > /dev/null; then
      addgroup -g $groupid $(printf %q "$groupname") && \
      adduser -u $userid -G $(printf %q "$groupname") -D -g '' $(printf %q "$USER")
    elif addgroup --help 2>&1 | grep -F -- '--gid ID' > /dev/null; then
      addgroup --gid $groupid $(printf %q "$groupname") && \
      adduser --uid $userid --gid $groupid --disabled-password --gecos '' $(printf %q "$USER")
    else
      echo 'error: Could not figure out how to add a new user.'
      false
    fi
  "
}

# dockerdev_start_new_dev_container <container-name> <image-name> \
#     [--x11] [--docker] [-- <docker-run-flags>...]
#   Start a new container with a certain image and set up a non-root user.
#   Options:
#   --x11     Connect the container to the X server on the host so that GUIs
#             can be used.
#   --docker  Connect the container to the Docker daemon on the host so that
#             it can use the Docker client.
dockerdev_start_new_dev_container() {
  local container_name=
  local image_name=
  local x11=false
  local docker=false
  while [[ $# -gt 0 ]]; do
    case $1 in
      --x11) x11=true ;;
      --docker) docker=true ;;
      --) shift; break ;;
      *)
        if [[ ! $container_name ]]; then
          container_name=$1
        elif [[ ! $image_name ]]; then
          image_name=$1
        else
          echo "error: unrecognized argument to dockerdev_start_new_dev_container: $1" 1>&2 &&
          echo "       Did you forget to use -- with dockerdev_ensure_dev_container_started?" 1>&2 &&
          return 1
        fi
        ;;
    esac
    shift
  done
  # Add the docker group to the user so we have the proper permissions to run
  # Docker-in-Docker. Only supported for Linux for now.
  local docker_in_docker_args=() &&
  if $docker; then
    case "$OSTYPE" in
      darwin*)
        echo 'error: --docker is not implemented for MacOS' 1>&2 &&
        return 1
        ;;
      *)
        local dockergroupid
        dockergroupid=$(getent group docker | cut -d : -f 3) &&
        docker_in_docker_args=( \
          --group-add "$dockergroupid" \
          -v /var/run/docker.sock:/var/run/docker.sock \
        )
        ;;
    esac
  fi
  if $x11; then
    local userid
    local groupid
    # Connect the container to the host's X server and make a home directory.
    # See http://wiki.ros.org/docker/Tutorials/GUI
    # and https://medium.com/@SaravSun/running-gui-applications-inside-docker-containers-83d65c0db110
    userid=$(id -u "$USER") &&
    groupid=$(id -g "$USER") &&
    case "$OSTYPE" in
      darwin*)
        # On Mac, follow this tutorial:
        # https://sourabhbajaj.com/blog/2017/02/07/gui-applications-docker-mac/
        local ip_address &&
        ip_address=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}') &&
        xhost + "$ip_address" > /dev/null &&
        dockerdev_start_new_container "$container_name" "$image_name" -it \
          -u "$userid":"$groupid" \
          ${docker_in_docker_args[@]+"${docker_in_docker_args[@]}"} \
          -e DISPLAY="$ip_address":0 \
          -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
          "$@" &&
        local add_user &&
        add_user=$(_dockerdev_add_user) &&
        docker exec -i -u 0:0 "$container_name" sh -c "$add_user"
        ;;
      *)
        dockerdev_start_new_container "$container_name" "$image_name" -it \
          -u "$userid":"$groupid" \
          ${docker_in_docker_args[@]+"${docker_in_docker_args[@]}"} \
          -e DISPLAY \
          -v /etc/group:/etc/group:ro \
          -v /etc/passwd:/etc/passwd:ro \
          -v /etc/shadow:/etc/shadow:ro \
          -v /etc/sudoers.d:/etc/sudoers.d:ro \
          -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
          -v "$HOME"/.Xauthority:"$HOME"/.Xauthority:rw \
          --net host \
          "$@" &&
          # See https://serverfault.com/questions/63764/create-home-directories-after-create-users/508509
          # Use `-u 0:0` to make sure we run as root.
          docker exec -i -u 0:0 "$container_name" sh -c "
            cd $HOME && \
            cp -r /etc/skel/. . && \
            chown -R $userid:$groupid . && \
            chmod -R go=u,go-w . && \
            chmod go= .
          "
        ;;
    esac
  else
    local add_user &&
    dockerdev_start_new_container \
      "$container_name" \
      "$image_name" \
      -it "$@" \
      ${docker_in_docker_args[@]+"${docker_in_docker_args[@]}"} &&
    # Add a user matching the user on the host system, so we can write files as
    # the same (non-root) user as the host. This allows us to do things like
    # write node_modules and lock files into a bind-mounted volume with the
    # correct permissions. We also need to set up a home directory for the user,
    # since some tools do not work right without one.
    add_user=$(_dockerdev_add_user) &&
    # Use `-u 0:0` to make sure we run as root.
    docker exec -i -u 0:0 "$container_name" sh -c "$add_user"
  fi
}

# dockerdev_ensure_container_started <image-name> [--on-start <command>] \
#     [-- <docker-run-flags>...]
#   Start a container with a certain image if it is not already running.
#   Optionally accepts the name of a command to be called just after the
#   container is started (--on-start). It will not be called if the container
#   is already running.
dockerdev_ensure_container_started() {
  _dockerdev_ensure_container_started_impl \
    --start dockerdev_start_new_container "$@"
}

# dockerdev_ensure_dev_container_started <image-name> [--on-start <command>] \
#     [--x11] [-- <docker-run-flags>...]
#   Start a container with a certain image if it is not already running, and
#   ensure that the container has a user that can write to the host system as
#   a non-root user, which is useful when using package manager tools like NPM
#   and Composer from inside the container. This is the function you should
#   use to create a container for your development environment.
#   Optionally accepts the name of a command to be called just after the
#   container is started (--on-start). It will not be called if the container
#   is already running. The command will receive one argument: the name of the
#   container.
#   If --x11 is used, then the X server of the host will be made accessible to
#   the container so that GUI programs can be used.
dockerdev_ensure_dev_container_started() {
  _dockerdev_ensure_container_started_impl \
    --start dockerdev_start_new_dev_container "$@"
}

# _dockerdev_ensure_container_started_impl <image-name> [--start <command>] \
#   [--on-start <command>] [<start-args>...] [-- <docker-run-flags>...]
_dockerdev_ensure_container_started_impl() {
  local start_container_cmd=
  local on_start_cmd=:
  local image_name=
  local start_args=()
  while [[ $# -gt 0 ]]; do
    case $1 in
      --start) shift; start_container_cmd=$1 ;;
      --on-start) shift; on_start_cmd=$1 ;;
      --) break ;;
      *)
        if [[ ! $image_name ]]; then
          image_name=$1
        else
          start_args+=("$1")
        fi
        ;;
    esac
    shift
  done
  local container_name=$image_name
  local info
  info=$(dockerdev_container_info "$container_name") &&
  if [[ $info =~ ^([^ ]+)\ (.+)$ ]]; then
    # A container with the right name exists.
    local container_image=${BASH_REMATCH[1]}
    local container_status=${BASH_REMATCH[2]}
    echo "Found container $container_name" 1>&2
    echo "  Image:  $container_image" 1>&2
    echo "  Status: $container_status" 1>&2
    if [[ $container_image = $image_name ]]; then
      # The container is running the correct image.
      if [[ $container_status =~ ^Up\  ]]; then
        # The container is running.
        echo "Container $container_name is already running" 1>&2
      else
        # The container needs to be started.
        echo "Starting container $container_name" 1>&2
        docker start "$container_name"
      fi
    else
      # The container is running a different image (i.e. one that is out of
      # date).
      local new_name="${container_name}_${container_image}"
      # The existing container needs to be stopped because it is most likely
      # bound to the same port we need.
      echo "Stopping container $container_name" 1>&2 &&
      docker stop "$container_name" &&
      echo "Renaming container $container_name to $new_name" 1>&2 &&
      docker rename "$container_name" "$new_name" &&
      $start_container_cmd ${start_args[@]+"${start_args[@]}"} "$image_name" "$container_name" "$@" &&
      $on_start_cmd "$container_name"
    fi
  else
    # No container with the expected name exists.
    $start_container_cmd ${start_args[@]+"${start_args[@]}"} "$image_name" "$container_name" "$@" &&
    $on_start_cmd "$container_name"
  fi
}

# dockerdev_run_in_container <docker-exec-args>...
#   Run a command in a container.
dockerdev_run_in_container() {
  local cols
  local lines
  local flags=i
  # Automatically use a pseduo-TTY if stdout is a TTY. Disabling this option
  # when stdout is not a TTY lets Docker play nice with batch submission
  # systems.
  if [[ -t 1 ]]; then
    flags+=t
  fi &&
  # COLUMNS and LINES fix an issue with terminal width.
  # See https://github.com/moby/moby/issues/33794
  cols=$(tput cols) &&
  lines=$(tput lines) &&
  docker exec -"$flags" -e COLUMNS="$cols" -e LINES="$lines" "$@"
}

# dockerdev_run_in_dev_container <docker-exec-args>...
#   Run a command in a container as the same user as the host user.
dockerdev_run_in_dev_container() {
  local userid
  local groupid
  userid=$(id -u "$USER") &&
  groupid=$(id -g "$USER") &&
  # -u lets us execute the command as the host user.
  dockerdev_run_in_container -u "$userid":"$groupid" "$@"
}

# dockerdev_get_service_container_name <service>
#   Get the full name of a service's container.
dockerdev_get_service_container_name() {
  local service=$1
  local id
  local name
  id=$(docker service ps "$service" --filter desired-state=running -q) &&
  [[ $id ]] &&
  name=$(docker service ps "$service" --filter id="$id" --format '{{.Name}}') &&
  printf '%s' "$name"."$id"
}

# dockerdev_get_container_image <container>
#   Get the full name of a container's image.
dockerdev_get_container_image() {
  local container=$1
  docker ps -a --filter name="^/$container\$" --format '{{.Image}}'
}

# dockerdev_ensure_service_image_updated <service> <image>
#   Ensure that a service is running a certain image. This is usually used to
#   ensure that the service is running the latest version of an image.
dockerdev_ensure_service_image_updated() {
  local service=$1
  local image=$2
  local container
  local current_image
  while ! container=$(dockerdev_get_service_container_name "$service"); do
    echo "Waiting for service $service to appear..." 1>&2 &&
    sleep 1
  done &&
  current_image=$(dockerdev_get_container_image "$container") &&
  while [[ $current_image = '' ]]; do
    echo "Waiting for container $container to appear..." 1>&2 &&
    sleep 1 &&
    { current_image=$(dockerdev_get_container_image "$container") || return; }
  done &&
  echo "Found container $container." 1>&2 &&
  if [[ $current_image != $image ]]; then
    echo "Updating image of service $service from $current_image to $image..." 1>&2 &&
    docker service update --image "$image" --force "$service" 1>&2
  else
    echo "Image of service $service is already up to date." 1>&2
  fi
}

# dockerdev_ping_container <container>
#   Tell whether a container is ready to execute commands.
dockerdev_ping_container() {
  local container=$1
  docker exec "$container" echo ready &> /dev/null
}

# dockerdev_ensure_dev_user_added <container>
#   Ensure that a non-root user matching the host user has been added to a
#   container. See also `dockerdev_start_new_dev_container`.
dockerdev_ensure_dev_user_added() {
  local container_name=$1
  local add_user
  # See comments in `dockerdev_start_new_dev_container`.
  # Check if the user has already been created. If not, create it.
  add_user=$(_dockerdev_add_user) &&
  docker exec -i -u 0:0 "$container_name" sh -c "
    if ! id -u $USER > /dev/null 2>&1; then
      $add_user
    fi
  "
}

# dockerdev_ensure_stack_container_ready <docker-compose-file> \
#     <stack-name> <service> [<image>]
#   Ensure that a stack has been deployed and that one of its services has a
#   container that is ready to receive commands. If an image argument is
#   given, ensure that the container is running that exact version of the
#   image. The stack, service, and container will be created if they do not
#   already exist, and the image will be updated if necessary.
dockerdev_ensure_stack_container_ready() {
  local compose_file=$1
  local stack=$2
  local service=$3
  local image=$4
  local full_service_name="$stack"_"$service"
  local container
  docker stack deploy -c "$compose_file" --prune "$stack" 1>&2 &&
  if [[ $image ]]; then
    dockerdev_ensure_service_image_updated "$full_service_name" "$image"
  fi &&
  container=$(dockerdev_get_service_container_name "$full_service_name") &&
  while ! dockerdev_ping_container "$container"; do
    echo "Polling container $container..." 1>&2 &&
    sleep 1
  done &&
  printf '%s' "$container"
}

# dockerdev_run_in_stack_container <docker-compose-file> <stack-name> \
#     <service> <image> <docker-exec-args>...
#   Ensure that a stack, service, and container have been created as in
#   `dockerdev_ensure_stack_container_ready`, and also run a command in
#   the container.
dockerdev_run_in_stack_container() {
  local compose_file=$1
  local stack=$2
  local service=$3
  local image=$4
  shift 4
  local container
  container=$(dockerdev_ensure_stack_container_ready \
    "$compose_file" "$stack" "$service" "$image") &&
  dockerdev_run_in_container "$container" "$@"
}

# dockerdev_run_in_dev_stack_container <docker-compose-file> <stack-name> \
#     <service> <image> <docker-exec-args>...
#   Ensure that a stack, service, and container have been created as in
#   `dockerdev_ensure_stack_container_ready`. Also create a non-root user in
#   the container and run a command in the container as that user.
dockerdev_run_in_dev_stack_container() {
  local compose_file=$1
  local stack=$2
  local service=$3
  local image=$4
  shift 4
  local container
  container=$(dockerdev_ensure_stack_container_ready \
    "$compose_file" "$stack" "$service" "$image") &&
  dockerdev_ensure_dev_user_added "$container" &&
  dockerdev_run_in_dev_container "$container" "$@"
}

# End of include guard.
fi
