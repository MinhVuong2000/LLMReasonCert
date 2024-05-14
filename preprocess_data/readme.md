How to create Subgraphs from SPARQL
---

### A. Firstly, you need to construct Freebase Database before querying via SparQL
1. Download [virtuoso-opensource](https://github.com/openlink/virtuoso-opensource/releases/tag/v7.2.11)
-> Extract and put the folder in the `./virtuoso_db`
2. Download and [virtuoso_db file](virtuoso_db)
-> unzip and  put the folder in the `./virtuoso_db/virtuoso_db`
   
Read more in [this link](https://juejin.cn/post/7283690681175113740)

### B. After finishing the preparation
It requires a terminal multiplexer, might use `vim` or `tmux`/`smux` (preferred). 
##### Window1. 
1. Change the working dir: `cd ./virtuoso_db`
2. `python3 virtuoso.py start 3001 -d virtuoso_db`\
To stop: `python3 virtuoso.py stop 3001`

##### Window2. 
Obtain a **raw-subgraph** via `SPARQL` and `CONSTRUCT` query and \
**subgraph** from the **raw-subgraph** by skipping unnamed entities:`h,r1,*`,`*,r2,t` ->`h,r1/r2,t`
- Handle CWQ: `python ./preprocess_data/cwq_graph.py`
- Handle GrailQA: `python ./preprocess_data/grailqa_graph.py`
