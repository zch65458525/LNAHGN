## LNAHGN
Codes for ICONIP2025 paper submission "LNAHGN: LLM-Guided Neighbor Aggregation for Heterogeneous Graph Neural Network"

## Requirements
python3.10

pytorch

dgl2.4.0

## Run
We have uploaded the LLM generated results for `ACM` dataset, you can see them in `/hgb/cache_data/ACM`. If you want to regenerate these results, change the models and API ley in `api.py` to your own.
To run the LNAHGN model, you should
```bash
cd hgb
python main.py --num-hops 4
```
