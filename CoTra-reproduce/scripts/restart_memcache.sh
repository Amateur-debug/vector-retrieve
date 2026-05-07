#!/bin/bash

resolve_hostname() {
  local hostname=$1
  if command -v getent &> /dev/null; then
    local ips
    ips=$(getent ahostsv4 "$hostname" | awk '/STREAM/ {print $1}')
  elif command -v host &> /dev/null; then
    local ips
    ips=$(host "$hostname" | awk '/has address/ {print $4}')
  else
    echo "Error: failed to find getent or host.. "
    return 1
  fi

  if [ -z "$ips" ]; then
    echo "failed to resolve: $hostname"
    return 1
  fi

  # pass lo
  local filtered_ips=()
  for ip in $ips; do
    if [[ $ip != 127.* ]]; then
      filtered_ips+=("$ip")
    fi
  done

  if [ ${#filtered_ips[@]} -eq 0 ]; then
    echo "can not find address"
    return 1
  fi

  echo "${filtered_ips[0]}"
  return 0
}


clear_memcache() {
  local CONF_FILE=$1
  if [ -f "$CONF_FILE" ]; then
    DOMAIN=$(sed -n '1p' "$CONF_FILE")  # read first line
    PORT=$(sed -n '2p' "$CONF_FILE")    # read second line
  else
    echo "config file $CONF_FILE does not exist."
    exit 1
  fi

  leader_hostname="$DOMAIN"
  if read_ip=$(resolve_hostname "$leader_hostname"); then
    echo "leader hostname $leader_hostname ip: $read_ip"
  else
    echo "failed to resolve leader machine ip"
  fi

  hostname=$(hostname)
  local_ip=$(resolve_hostname "$hostname")
  echo "local hostname: $hostname ip: $local_ip"

  if [[ "$read_ip" == "$local_ip" ]]; then
    # restart memcache
    addr=$DOMAIN
    port=$PORT

    # kill old me
    if [[ -f /tmp/memcached.pid ]]; then
        kill "$(cat /tmp/memcached.pid)" 2>/dev/null
        rm -f /tmp/memcached.pid
    fi

    # launch memcached
    memcached -u root -l ${addr} -p  ${port} -c 10000 -d -P /tmp/memcached.pid
    sleep 1

    # init 
    echo -e "set serverNum 0 0 1\r\n0\r\nquit\r" | nc ${addr} ${port}
    echo -e "set clientNum 0 0 1\r\n0\r\nquit\r" | nc ${addr} ${port}
    echo "memcache clear and restart"
  fi
  return 1
}





