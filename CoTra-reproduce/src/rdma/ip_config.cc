#include "rdma/ip_config.h"

// Get local machine ip addr.
std::string get_local_ip_addr() {
  struct ifaddrs *ifaddr, *ifa;
  char ip[INET_ADDRSTRLEN];

  if (getifaddrs(&ifaddr) == -1) {
    perror("getifaddrs");
    return "";
  }

  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (ifa->ifa_addr->sa_family == AF_INET) {  // IPv4
      struct sockaddr_in *sa = (struct sockaddr_in *)ifa->ifa_addr;
      inet_ntop(AF_INET, &(sa->sin_addr), ip, INET_ADDRSTRLEN);
      std::string ifaceName(ifa->ifa_name);
      if (ifaceName != "lo") {
        freeifaddrs(ifaddr);
        return std::string(ip);
      }
    }
  }

  freeifaddrs(ifaddr);
  return "";
}

// DNS
std::string resolve_hostname(const std::string &hostname) {
  struct addrinfo hints {}, *res;
  hints.ai_family = AF_INET;  // IPv4 only
  hints.ai_socktype = SOCK_STREAM;

  // 解析主机名
  if (getaddrinfo(hostname.c_str(), nullptr, &hints, &res) != 0) {
    std::cerr << "getaddrinfo failed for hostname: " << hostname << std::endl;
    return "";
  }

  std::string result;
  for (struct addrinfo *p = res; p != nullptr; p = p->ai_next) {
    char ip[INET_ADDRSTRLEN];
    auto *addr = (struct sockaddr_in *)p->ai_addr;
    inet_ntop(AF_INET, &(addr->sin_addr), ip, INET_ADDRSTRLEN);

    // pass 127.0.0.1
    if (std::string(ip).find("127.") != 0) {
      result = ip;
      break;  
    }
  }

  freeaddrinfo(res);

  if (result.empty()) {
      std::cerr << "No non-loopback address found for hostname: " << hostname << std::endl;
  }

  return result;
}


// read ip_list
std::unordered_map<std::string, int> read_ip_map(const std::string &filename) {
  std::unordered_map<std::string, int> ipMap;
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Fatal error: Can not open ip_list file: " << filename
              << std::endl;
    return ipMap;
  }

  // get first line of leader machine info. 
  std::string addr, port;
  std::getline(file, addr);
  std::getline(file, port);

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string hostOrIP;
    int id;
    if (std::getline(iss, hostOrIP, '=') && iss >> id) {
      std::string ip = resolve_hostname(hostOrIP);
      if (!ip.empty()) {
        ipMap[ip] = id;
      } else {
        std::cerr << "WARN: Can not resolve" << hostOrIP << ", skip."
                  << std::endl;
      }
    }
  }

  file.close();
  return ipMap;
}

int get_machine_id(std::string config_file) {
  // const std::string configFile = "../scripts/ip_list.conf";

  auto ipMap = read_ip_map(config_file);

  if (ipMap.empty()) {
    std::cerr << "Error: ip_list file is empty or cannot resolve " << std::endl;
    return 1;
  }

  // Get local machine IP addr.
  std::string localIP = get_local_ip_addr();
  if (localIP.empty()) {
    std::cerr << "Can not get local IP addr." << std::endl;
    return 1;
  }

  std::cout << "Local machine IP addr: " << localIP << std::endl;

  // find ID of IP addr.
  auto it = ipMap.find(localIP);
  int machine_id = -1;
  if (it != ipMap.end()) {
    machine_id = it->second;
    std::cout << "Local machine ID: " << machine_id << std::endl;
  } else {
    std::cout << "Error: local machine IP cannot find in list." << std::endl;
    abort();
  }

  return machine_id;
}


std::vector<std::string> get_machine_name(std::string config_file){
  std::vector<std::string> machine_name;
  std::ifstream file(config_file);

  if (!file.is_open()) {
    std::cerr << "Fatal error: Can not open ip_list file: " << config_file
              << std::endl;
    return machine_name;
  }

  // get first line of leader machine info. 
  std::string addr, port;
  std::getline(file, addr);
  std::getline(file, port);

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string hostOrIP;
    int id;
    if (std::getline(iss, hostOrIP, '=') && iss >> id) {
      machine_name.push_back(hostOrIP);
    }
  }

  file.close();
  return machine_name;
}

