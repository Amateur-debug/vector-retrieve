#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <queue>
#include <vector>

class GraphReorder {
private:
  uint32_t vertexNum = 0;
  uint32_t edgeNum = 0;

  uint32_t *csr_offset = nullptr;
  uint32_t *csr_dest = nullptr;
  uint32_t *csr_weight = nullptr;

  uint32_t *outDegree = nullptr;
  uint32_t *inDegree = nullptr;

private:
  struct Vid_degree_type {
    uint32_t vertex_id;
    uint32_t degree;
  };
  std::vector<Vid_degree_type> sort_vec;

  std::vector<uint32_t> order;
  std::vector<uint32_t> additionInfo;

public:
  /* *********************************************************************************************************
   * @description: CGgraph Reorder
   * @param [uint32_t] vertexNum_     The number of vertices
   * @param [uint32_t] edgeNum_       The number of edges
   * @param [uint32_t] *csr_offset_   The offset of CSR
   * @param [uint32_t] *csr_dest_     The dest of CSR
   * @param [uint32_t] *csr_weight_   The edge weight of CSR
   * @param [uint32_t] *outDegree_    The out-going of verices
   * @param [uint32_t] *inDegree_     The in-incoming of vertices
   * @return [*]
   * *********************************************************************************************************/
  GraphReorder(uint32_t vertexNum_, uint32_t edgeNum_, uint32_t *csr_offset_,
               uint32_t *csr_dest_, uint32_t *csr_weight_, uint32_t *outDegree_,
               uint32_t *inDegree_) {
    vertexNum = vertexNum_;
    edgeNum = edgeNum_;
    csr_offset = csr_offset_;
    csr_dest = csr_dest_;
    csr_weight = csr_weight_;
    outDegree = outDegree_;
    inDegree = inDegree_;

    generateReorder();
  }

  ~GraphReorder() {
    if (csr_offset != nullptr) {
      delete[] csr_offset;
      csr_offset = nullptr;
    }

    if (csr_dest != nullptr) {
      delete[] csr_dest;
      csr_dest = nullptr;
    }

    if (csr_weight != nullptr) {
      delete[] csr_offset;
      csr_weight = nullptr;
    }

    if (outDegree != nullptr) {
      delete[] outDegree;
      outDegree = nullptr;
    }

    if (inDegree != nullptr) {
      delete[] inDegree;
      inDegree = nullptr;
    }
  }

  /**********************************************************************************************************
   * @description: The result of reorder
   *               For example: when a 8 vertices graph return the vector:
   *                    old:    [0] [1] [2] [3] [4] [5] [6] [7]
   *                    new:    <0> <1> <7> <6> <5> <3> <4> <2>
   *               It shows:
   *               The vertex with the original Id of [0] is assigned a new Id
   *<0> The vertex with the original Id of [1] is assigned a new Id <1> The
   *vertex with the original Id of [2] is assigned a new Id <7>
   *               ...
   * @return [*]
   **********************************************************************************************************/
  std::vector<uint32_t> getOrder() { return order; }

  /**********************************************************************************************************
   * @description: The number of sink vertices
   * @return [*]
   **********************************************************************************************************/
  uint32_t getAdditionInfo() { return additionInfo[0]; }

private:
  void generateReorder() {
    sort_vec.resize(vertexNum);
    sortDegree();
    printf("Finished Sort degree \n");

    CGgraphR(order, additionInfo);
    printf("CGgraphR zeroDegreenNum = %u (%.2f%%)\n", additionInfo[0],
           ((double)additionInfo[0] / vertexNum) * 100);
  }

  void sortDegree() {
#pragma omp parallel for
    for (uint32_t vertexId = 0; vertexId < vertexNum; vertexId++) {
      Vid_degree_type vid_degree;
      vid_degree.vertex_id = vertexId;
      vid_degree.degree = inDegree[vertexId];
      sort_vec[vertexId] = vid_degree;
    }

    std::sort(sort_vec.begin(), sort_vec.end(),
              [&](Vid_degree_type &a, Vid_degree_type &b) -> bool {
                if (a.degree > b.degree)
                  return true;
                else if (a.degree == b.degree) {
                  if (a.vertex_id < b.vertex_id)
                    return true;
                  else
                    return false;
                } else
                  return false;
              });
  }

  void CGgraphR(std::vector<uint32_t> &retorder,
                std::vector<uint32_t> &additionInfo) {
    std::queue<uint32_t> que;
    bool *BFSflag = new bool[vertexNum];
    memset(BFSflag, 0, sizeof(bool) * vertexNum);

    std::vector<uint32_t> tmp;
    uint32_t now;
    std::vector<uint32_t> order;
    std::vector<uint32_t> zeroDegree;
    for (uint32_t k = 0; k < vertexNum; k++) {
      uint32_t temp = sort_vec[k].vertex_id;
      if (BFSflag[temp] == false) {
        if (outDegree[temp] == 0) {
          zeroDegree.push_back(temp);
          BFSflag[temp] = true;
        } else {
          que.push(temp);
          BFSflag[temp] = true;
          order.push_back(temp);
        }

        while (que.empty() == false) {
          now = que.front();
          que.pop();
          tmp.clear();
          for (uint32_t it = csr_offset[now], limit = csr_offset[now + 1];
               it < limit; it++) {
            uint32_t dest = csr_dest[it];

            if ((outDegree[dest] == 0) && (BFSflag[dest] == false)) {
              zeroDegree.push_back(dest);
              BFSflag[dest] = true;
            } else {
              tmp.push_back(dest);
            }
          }
          sort(tmp.begin(), tmp.end(),
               [&](const uint32_t &a, const uint32_t &b) -> bool {
                 if (inDegree[a] > inDegree[b])
                   return true;
                 else
                   return false;
               });

          for (uint32_t nbrId = 0; nbrId < tmp.size(); nbrId++) {
            if (BFSflag[tmp[nbrId]] == false) {
              que.push(tmp[nbrId]);
              BFSflag[tmp[nbrId]] = true;
              order.push_back(tmp[nbrId]);
            }
          }
        }
      }
    }

    delete[] BFSflag;
    if ((order.size() + zeroDegree.size()) != vertexNum) {
      printf("((order.size() + zeroDegree.size()) != vertexNum), order.size() "
             "= %zu, zeroDegree.size() = %zu",
             order.size(), zeroDegree.size());
    }

    additionInfo.resize(1);
    additionInfo[0] = zeroDegree.size();
    printf("additionInfo[0] = %zu", static_cast<uint64_t>(additionInfo[0]));

    retorder.resize(vertexNum);
    retorder.insert(retorder.begin(), order.begin(), order.end());
    retorder.insert(retorder.begin() + order.size(), zeroDegree.begin(),
                    zeroDegree.end());
  }
};

/*

CGgraph_Reorder *reorder = new CGgraph_Reorder(csrResult.vertexNum,
   csrResult.edgeNum, csrResult.csr_offset, csrResult.csr_dest,
                                                   csrResult.csr_weight,
   csrResult.outDegree, csrResult.inDegree); std::vector<uint32_t> result =
   reorder->getOrder();

    std::stringstream ss;
    for (uint32_t vertexId = 0; vertexId < csrResult.vertexNum; vertexId++)
    {
        ss << "[" << vertexId << "]:" << result[vertexId] << " ";
    }
    Msg_info("%s", ss.str().c_str());
    Msg_info("The num of sink vertices = %u", reorder->getAdditionInfo());
    return;
*/