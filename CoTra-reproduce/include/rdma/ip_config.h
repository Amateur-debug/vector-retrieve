#pragma once

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

// Get local machine ip addr.
std::string get_local_ip_addr();
// DNS
std::string resolve_hostname(const std::string &hostname);
// read ip_list
std::unordered_map<std::string, int> read_ip_map(const std::string &filename);

int get_machine_id(std::string config_file);

std::vector<std::string> get_machine_name(std::string config_file);
